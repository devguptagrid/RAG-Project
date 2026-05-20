from pyexpat import model
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from vertexai.preview.generative_models import GenerativeModel
import vertexai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.models import PointStruct
import numpy as np
from sentence_transformers import CrossEncoder
import pdfplumber

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
vertexai.init(project="gd-gcp-gridu-genai", location="us-central1")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    
    text = ""
    for page in doc:
        page_text = page.get_text()
        
        # Basic cleaning
        if page_text:
            text += page_text
    
    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # slightly bigger for better context
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    return chunks



def filter_chunks(chunks):
    filtered = []
    
    for chunk in chunks:
        chunk = chunk.strip()
        
        # 1. Remove very small chunks
        if len(chunk) < 80:
            continue
        
        # 2. Remove chunks with too many numbers (likely tables/TOC)
        num_ratio = sum(c.isdigit() for c in chunk) / len(chunk)
        if num_ratio > 0.3:
            continue
        
        # 3. Remove common noisy patterns
        if re.search(r"Table of Contents|IFC 2024 ANNUAL REPORT|page \d+", chunk, re.IGNORECASE):
            continue
        
        # 4. Remove chunks that are mostly uppercase (titles)
        if chunk.isupper():
            continue
        
        filtered.append(chunk)
    
    return filtered


def create_embeddings(model,chunks):
    
    
    embeddings = model.encode(chunks, normalize_embeddings=True)
    
    return embeddings


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatIP(dimension)  # simple distance index
    index.add(embeddings)
    
    return index

def search_text(query, model, index, chunks, k=10):
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, k)
    
    return [chunks[i] for i in indices[0]]   # ✅ returns text

def search_indices(query, model, index, chunks, k=10):
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, k)
    
    return indices[0]   # ✅ returns indices

def cross_rerank(query, retrieved_chunks, k=3):
    
    pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]
    
    scores = cross_encoder.predict(pairs)
    
    scored = list(zip(scores, retrieved_chunks))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    
    return [chunk for _, chunk in scored[:k]]


def generate_answer(query, context_chunks):
    model = GenerativeModel("gemini-2.0-flash")
    
    context = "\n\n".join(context_chunks)
    
    prompt = f"""
    You are given context which may include both text and tables.

    Instructions:
    - If the context contains tables, carefully read them.
    - Extract exact values from tables when needed.
    - Use table data for numerical questions.
    - If multiple years are present, compare them correctly.
    - Do NOT ignore tables.

    Context:
    {context}

    Question:
    {query}

    Answer clearly and accurately.
    """
    
    response = model.generate_content(prompt)
    
    return response.text


def create_qdrant_client():
    client = QdrantClient(":memory:")  # local in-memory DB
    return client

def create_collection(client, collection_name, vector_size):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

def insert_into_qdrant(client, collection_name, chunks, embeddings):
    points = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={"text": chunk}
            )
        )
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )

def search_qdrant(client, collection_name, query, model, k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=k
    )
    
    return [res.payload["text"] for res in results.points]


def chunk_text_with_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    
    chunks = []
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,       # slightly bigger for better context
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        split_chunks = splitter.split_text(page_text)
        
        for chunk in split_chunks:
            chunks.append({
                "text": chunk,
                "page": page_num + 1,
                "type": "text"
            })
    
    return chunks

def filter_by_page(chunks, start, end):
    return [c for c in chunks if start <= c["page"] <= end]


def run_phase3_pipeline(query, model, index, metadata_chunks, start_page=None, end_page=None):

    print("USING NEW PIPELINE")

    # 🔹 Step 1: Prepare text list (same used for embeddings)
    texts = [c["text"] for c in metadata_chunks]

    # 🔹 Step 2: Retrieve indices (NOT text)
    indices = search_indices(query, model, index, texts, k=15)

    # 🔹 Step 3: Map indices → metadata chunks (direct, no matching)
    retrieved_chunks = [metadata_chunks[i] for i in indices]

    # 🔹 Step 4: Apply metadata filtering (optional)
    if start_page is not None and end_page is not None:
        retrieved_chunks = filter_by_page(retrieved_chunks, start_page, end_page)

    # 🔹 Step 5: Cross-encoder reranking
    reranked_chunks = cross_rerank(query, retrieved_chunks, k=7)

    # 🔹 Step 6: Extract final context
    final_contexts = [c["text"] for c in reranked_chunks]

    # 🔹 Debug (optional but useful)
    print("\nFINAL CONTEXT:\n")
    for c in final_contexts:
        print(c[:200])
        print("------")

    # 🔹 Step 7: Generate answer
    answer = generate_answer(query, final_contexts)

    return {
        "answer": answer,
        "retrieved_chunks": reranked_chunks
    }





def extract_tables(pdf_path):
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            extracted_tables = page.extract_tables()

            if extracted_tables:
                for table in extracted_tables:
                    tables.append({
                        "table": table,
                        "page": page_num + 1
                    })

    return tables

def table_to_text(table):
    rows = []
    
    for row in table:
        # remove empty cells
        cleaned_row = [cell.strip() for cell in row if cell and cell.strip() != ""]
        
        if cleaned_row:
            rows.append(" | ".join(cleaned_row))
    
    return "\n".join(rows)

def create_table_chunks(tables):
    table_chunks = []

    for t in tables:
        table_text = table_to_text(t["table"])

        if len(table_text.strip()) == 0:
            continue

        table_chunks.append({
            "text": table_text,
            "type": "table",
            "page": t["page"]
        })

    return table_chunks

def score_table(chunk, query):
    text = chunk["text"].lower()
    query_words = query.lower().split()

    score = 0

    # 1. keyword overlap (general)
    for word in query_words:
        if word in text:
            score += 3

    # 2. boost for multi-word matches
    query_phrase = " ".join(query_words)
    if query_phrase in text:
        score += 10

    # 3. numeric relevance (important for tables)
    import re
    numbers = re.findall(r"\d+", text)
    if len(numbers) >= 3:
        score += 5

    # 4. currency / financial signals
    if "$" in text:
        score += 3

    return score
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

def search(query,model, index, chunks, k=3):
    
    query_embedding = model.encode([query],normalize_embeddings=True)
    
    distances, indices = index.search(query_embedding, k)
    
    results = [chunks[i] for i in indices[0]]
    
    return results


def rerank(query, retrieved_chunks, model, chunk_embeddings, all_chunks):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    
    scored = []
    
    for chunk in retrieved_chunks:
        idx = all_chunks.index(chunk)
        chunk_embedding = chunk_embeddings[idx]
        
        score = np.dot(query_embedding, chunk_embedding)
        scored.append((score, chunk))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    
    return [chunk for _, chunk in scored]

def generate_answer(query, context_chunks):
    model = GenerativeModel("gemini-2.0-flash")
    
    context = "\n\n".join(context_chunks)
    
    prompt = f"""
    Answer the question based ONLY on the context below.

    Context:
    {context}

    Question:
    {query}

    If the answer is not present, say "Not found in document".
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
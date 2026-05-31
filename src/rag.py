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
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration

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
    You are given context from a financial report.

    IMPORTANT INSTRUCTIONS:
    - Data may come from tables converted to text.
    - Carefully align values with their correct labels (years, categories).
    - Do NOT guess.
    - If multiple percentages are present, match the correct year explicitly.
    - Prefer explicitly labeled values over ambiguous ones.

    Context:
    {context}

    Question:
    {query}

    Answer clearly and correctly.
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
    if not table or len(table) < 2:
        return ""

    headers = table[0]  # first row

    lines = []

    for row in table[1:]:
        row_text = []

        for i in range(len(row)):
            if row[i] and headers[i]:
                row_text.append(f"{headers[i]}: {row[i]}")

        if row_text:
            lines.append(" | ".join(row_text))

    return "\n".join(lines)

def create_table_chunks(tables):
    chunks = []

    for t in tables:
        table = t["table"]
        page = t["page"]

        text = table_to_text(table)

        # 🔥 KEEP WHOLE TABLE AS ONE CHUNK
        chunks.append({
            "text": text,
            "type": "table",
            "page": page
        })

    return chunks

def score_table(chunk, query):
    text = chunk["text"].lower()
    query = query.lower()

    score = 0

    # 🔥 strong keyword matching
    keywords = [
        "equity", "investment", "portfolio",
        "percentage", "%", "share", "accounted"
    ]

    for k in keywords:
        if k in text and k in query:
            score += 5

    # 🔥 year alignment
    if "2023" in text and "2023" in query:
        score += 5

    if "2024" in text and "2024" in query:
        score += 3

    # 🔥 percentage presence
    if "%" in text:
        score += 3

    # 🔥 table title relevance
    if "portfolio" in text:
        score += 4

    return score

import fitz  # PyMuPDF
import os

def extract_images(pdf_path, output_dir="images"):
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"page{page_num+1}_img{img_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_name)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append({
                "path": image_path,
                "page": page_num + 1
            })

    return image_paths

def is_useful_image(description):
    keywords = ["chart", "graph", "bar", "trend", "figure", "values"]
    return any(k in description.lower() for k in keywords)



def describe_image(image_path):
    model = GenerativeModel("gemini-2.0-flash")

    # read image as bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_part = Part.from_data(
        data=image_bytes,
        mime_type="image/png"   # change if jpg
    )

    prompt = """
    You are analyzing a chart or graph from a financial report.

    Your task:
    - Identify what the image represents
    - Extract important values (numbers, years, categories)
    - Describe trends or comparisons
    - Be precise and structured

    Ignore decorative or irrelevant details.
    """

    response = model.generate_content([prompt, image_part])

    return response.text

def is_useful_image(desc):
    keywords = ["chart", "graph", "figure", "trend", "increase", "decrease", "values"]
    return any(k in desc.lower() for k in keywords)

def detect_figure_chunks(chunks):
    figure_chunks = []

    for c in chunks:
        text = c.lower()

        if "figure" in text or "chart" in text:
            figure_chunks.append({
                "text": c,
                "type": "image",   # treat as image
                "page": None
            })

    return figure_chunks


def pdf_to_images(pdf_path, output_dir="page_images"):
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) #PDF page → image (pixel map)
        path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(path)

        image_paths.append({
            "path": path,
            "page": i + 1
        })

    return image_paths

# from PIL import Image

def create_image_patches(image_path, page_number, patch_size=128):
    img = Image.open(image_path).convert("RGB")

    patches = []
    width, height = img.size

    for x in range(0, width, patch_size):
        for y in range(0, height, patch_size):

            patch = img.crop((
                x,
                y,
                min(x + patch_size, width),
                min(y + patch_size, height)
            ))

            patches.append({
                "image": patch,
                "page": page_number,
                "coords": (x, y),
                "source": image_path,
                "type": "patch"
            })

    return patches




def load_clip_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.to(device)

    return model, processor, device

# def embed_image_patches(model, processor, patches, device):
    
#     embeddings = []

#     for p in patches:
#         inputs = processor(
#             images=p["image"],
#             return_tensors="pt"
#         ).to(device)

#         with torch.no_grad():
#             image_features = model.get_image_features(**inputs)

#         # 🔥 FIX: ensure tensor
#         if hasattr(image_features, "pooler_output"):
#             image_features = image_features.pooler_output

#         # normalize embedding (VERY IMPORTANT)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#         embeddings.append(image_features.squeeze().cpu().numpy())

#     return embeddings

# def embed_query_clip(model, processor, query, device):
    

#     inputs = processor(
#         text=query,
#         return_tensors="pt",
#         padding=True
#     ).to(device)

#     with torch.no_grad():
#         text_features = model.get_text_features(**inputs)

#     # 🔥 FIX: ensure tensor
#         if hasattr(text_features, "pooler_output"):
#             text_features = text_features.pooler_output

#     # normalize (VERY IMPORTANT)
#     text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#     return text_features.squeeze().cpu().numpy()


# def retrieve_top_patches(query_embedding, patch_embeddings, patches, top_k=5):
    
#     scores = []

#     for i, patch_emb in enumerate(patch_embeddings):
#         score = np.dot(query_embedding, patch_emb)  # cosine similarity (since normalized)
#         scores.append((score, i))

#     # sort by score descending
#     scores.sort(reverse=True, key=lambda x: x[0])

#     top_indices = [idx for _, idx in scores[:top_k]]

#     retrieved_patches = [patches[i] for i in top_indices]

#     return retrieved_patches






def load_hybrid_models():
    # CLIP (you already have)
    clip_model, clip_processor, device = load_clip_model()

    # BLIP
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    # Text model
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    return clip_model, clip_processor, blip_model, blip_processor, text_model, device

def generate_patch_caption(blip_model, blip_processor, image, device):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = blip_model.generate(**inputs)

    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    return caption

def project_to_dim(vec, target_dim=512):
    if len(vec) == target_dim:
        return vec

    # simple projection (pad or truncate)
    if len(vec) < target_dim:
        pad = np.zeros(target_dim - len(vec))
        return np.concatenate([vec, pad])
    else:
        return vec[:target_dim]

def embed_patch_hybrid(
    clip_model,
    clip_processor,
    blip_model,
    blip_processor,
    text_model,
    patch,
    device
):
    image = patch["image"]

    # 🔹 CLIP embedding
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs)

# 🔥 FIX: ensure tensor
    if hasattr(img_emb, "pooler_output"):
        img_emb = img_emb.pooler_output

    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    img_emb = img_emb.squeeze().cpu().numpy()

    # 🔹 BLIP caption
    caption = generate_patch_caption(blip_model, blip_processor, image, device)

    # 🔹 Text embedding
    txt_emb = text_model.encode(caption)
    txt_emb = txt_emb / np.linalg.norm(txt_emb)

    txt_emb = project_to_dim(txt_emb, 512)
    # 🔹 Combine
    fused = 0.6 * img_emb + 0.4 * txt_emb
    fused = fused / np.linalg.norm(fused)

    return fused, caption

def embed_all_patches_hybrid(
    clip_model,
    clip_processor,
    blip_model,
    blip_processor,
    text_model,
    patches,
    device
):
    embeddings = []
    captions = []

    for i, p in enumerate(patches):
        print(f"Embedding patch {i}")

        emb, cap = embed_patch_hybrid(
            clip_model,
            clip_processor,
            blip_model,
            blip_processor,
            text_model,
            p,
            device
        )

        embeddings.append(emb)
        captions.append(cap)

    return embeddings, captions

def embed_query_hybrid(clip_model, clip_processor, text_model, query, device):
    # CLIP
    inputs = clip_processor(text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_emb = clip_model.get_text_features(**inputs)
# 🔥 FIX: ensure tensor
    if hasattr(clip_emb, "pooler_output"):
        clip_emb = clip_emb.pooler_output

    clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)
    clip_emb = clip_emb.squeeze().cpu().numpy()

    # TEXT
    text_emb = text_model.encode(query)
    text_emb = text_emb / np.linalg.norm(text_emb)
    text_emb = project_to_dim(text_emb, 512)
    # COMBINE
    fused = 0.5 * clip_emb + 0.5 * text_emb
    fused = fused / np.linalg.norm(fused)

    return fused

def extract_numbers(text):
    return re.findall(r"\d+", text)

def is_numeric_query(query):
    import re
    return bool(re.search(r"\d", query))

def retrieve_hybrid(query_emb, patch_embeddings, patches, query, captions, top_k=5):
    import numpy as np
    import re

    query_numbers = re.findall(r"\d+", query)
    query_words = query.lower().split()

    scores = []

    for i, emb in enumerate(patch_embeddings):
        score = np.dot(query_emb, emb)

        caption = captions[i].lower()
        page_text = patches[i].get("page_text", "")

        # 🔥 NUMBER BOOST (STRONG)
        for num in query_numbers:
            if num in page_text:
                score += 1.5   # strong boost

        # 🔥 KEYWORD BOOST
        for word in ["sector", "industry", "distribution"]:
            if word in page_text:
                score += 0.5

        # 🔥 CAPTION BOOST (weak)
        for word in query_words:
            if word in caption:
                score += 0.1

        scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])

    return [patches[i] for _, i in scores[:top_k]]

def attach_text_to_patches(patches, text_chunks):
    for p in patches:
        page = p["page"]

        page_text = " ".join(
            [c["text"] for c in text_chunks if c["page"] == page]
        )

        p["page_text"] = page_text.lower()

    return patches


def extract_answer_from_page(query, text_chunks, top_pages):
    import re

    query_number = re.findall(r"\d{1,3},?\d*", query)
    if not query_number:
        return "No number found"

    query_number = query_number[0].replace(",", "")

    for chunk in text_chunks:
        if chunk["page"] in top_pages:

            lines = [l.strip() for l in chunk["text"].split("\n") if l.strip()]

            for i in range(len(lines)):
                line_clean = lines[i].replace(",", "")

                if query_number in line_clean:

                    # 🔥 scan upward until we find a proper label
                    for j in range(i-1, -1, -1):
                        candidate = lines[j]

                        # skip numeric lines
                        if any(char.isdigit() for char in candidate):
                            continue

                        # skip headers
                        if "total" in candidate.lower():
                            continue

                        return candidate

    return "Not found"

def extract_text_answer(query, text_chunks, top_pages):
    
    query_words = query.lower().split()

    scored_chunks = []

    for c in text_chunks:
        if c["page"] in top_pages:
            score = sum(word in c["text"].lower() for word in query_words)
            scored_chunks.append((score, c["text"]))

    scored_chunks.sort(reverse=True)

    context = [c for _, c in scored_chunks[:8]]

    return generate_answer(query, context)

def get_final_answer(query, text_chunks, top_pages):
    
    if is_numeric_query(query):
        return extract_answer_from_page(query, text_chunks, top_pages)
    
    else:
        return extract_text_answer(query, text_chunks, top_pages)
    

def keyword_score(text, query):
    query_words = query.lower().split()
    text = text.lower()

    return sum(word in text for word in query_words)

def get_top_pages_hybrid(query, text_chunks, text_model, index, texts, k=5):
    
    # 🔹 FAISS retrieval
    indices = search_indices(query, text_model, index, texts, k=k)
    faiss_pages = [text_chunks[i]["page"] for i in indices]

    # 🔹 Keyword scoring
    page_scores = {}

    for chunk in text_chunks:
        page = chunk["page"]
        score = keyword_score(chunk["text"], query)

        if page not in page_scores:
            page_scores[page] = 0

        page_scores[page] += score

    # 🔹 Sort keyword pages
    keyword_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    keyword_pages = [p for p, _ in keyword_pages[:k]]

    # 🔥 COMBINE BOTH
    final_pages = list(set(faiss_pages + keyword_pages))

    return final_pages



####################################### paligemma



from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch

def load_paligemma_model():
    print("Loading PaliGemma...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        "google/paligemma-3b-pt-224"
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto"
    )

    model.to(device)

    return model, processor, device


def embed_patch_paligemma(model, processor, patch, device):
    image = patch["image"]

    # dummy prompt (required by model)
    prompt = "describe the image"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 🔥 take LAST hidden state
    hidden_states = outputs.hidden_states[-1]

    # 🔥 mean pooling (convert tokens → vector)
    embedding = hidden_states.mean(dim=1)

    # normalize
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.squeeze().cpu().numpy()

def embed_all_patches_paligemma(model, processor, patches, device, batch_size=2):
    import torch
    import numpy as np

    embeddings = []

    print(f"Total patches: {len(patches)}")

    for i in range(0, len(patches), batch_size):
        batch = patches[i:i + batch_size]

        print(f"Processing batch {i} → {i + len(batch)}")

        images = [p["image"] for p in batch]

        prompts = ["describe the image"] * len(images)

        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]  # (B, T, D)

        # 🔥 mean pooling
        batch_emb = hidden_states.mean(dim=1)

        # normalize
        batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)

        batch_emb = batch_emb.cpu().numpy()

        embeddings.extend(batch_emb)

        # 🔥 free memory (IMPORTANT for Mac)
        del inputs, outputs, hidden_states, batch_emb
        torch.mps.empty_cache() if device == "mps" else None

    return embeddings

def embed_patch_tokens_paligemma(model, processor, patch, device):
    import torch

    image = patch["image"]
    prompt = "<image> describe the image"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 🔥 LAST hidden state → (1, T, D)
    token_embeddings = outputs.hidden_states[-1].squeeze(0)  # (T, D)

    # normalize EACH token
    token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)

    return token_embeddings.cpu().numpy()

def maxsim_score_full(query_tokens, patch_tokens):
    import numpy as np

    # query_tokens → (Q, D)
    # patch_tokens → (P, D)

    # similarity matrix → (Q, P)
    sim_matrix = np.matmul(query_tokens, patch_tokens.T)

    # 🔥 for each query token → best matching patch token
    max_sim_per_query = sim_matrix.max(axis=1)  # (Q,)

    # 🔥 final score = sum of max similarities
    return max_sim_per_query.sum()

def retrieve_patches_maxsim_paligemma(
    query_tokens,
    patch_token_embeddings,
    patches,
    top_k=5
):
    scores = []

    for i, patch_tokens in enumerate(patch_token_embeddings):

        score = maxsim_score_full(query_tokens, patch_tokens)

        scores.append((score, i))

    # sort descending
    scores.sort(reverse=True, key=lambda x: x[0])

    # return top patches
    return [patches[i] for _, i in scores[:top_k]]


# def embed_query_tokens_paligemma(model, processor, query, device):
#     import torch

#     inputs = processor(
#         text=query,
#         return_tensors="pt"
#     ).to(device)

#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)

#     query_tokens = outputs.hidden_states[-1].squeeze(0)  # (Q, D)

#     query_tokens = query_tokens / query_tokens.norm(dim=-1, keepdim=True)

#     return query_tokens.cpu().numpy()

def generate_answer_paligemma(model, processor, query, best_page, images, top_pages, device):
    import torch

    # 🔥 USE HYBRID PAGE (NOT MAXSIM)
    page_number = best_page

    image = get_page_image(images, page_number)

    prompt = f"<image> Extract the exact answer from the document. Question: {query}"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120
        )

    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    return answer

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel
import torch

def load_text_token_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

def embed_query_tokens_text(query, tokenizer, model):
    import torch
    import numpy as np

    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # (1, T, D) → (T, D)
    token_embeddings = outputs.last_hidden_state.squeeze(0)

    # normalize
    token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)

    return token_embeddings.numpy()

def project_query_to_patch_dim(query_tokens, target_dim=2048):
    """
    Project query tokens (384 → 2048)
    """

    current_dim = query_tokens.shape[1]

    # simple linear projection (random)
    projection_matrix = np.random.randn(current_dim, target_dim)

    projected = np.matmul(query_tokens, projection_matrix)

    # normalize
    projected = projected / np.linalg.norm(projected, axis=1, keepdims=True)

    return projected

from PIL import Image

def get_page_image(images, page_number):
    for img in images:
        if img["page"] == page_number:
            return Image.open(img["path"]).convert("RGB")
    return None






###########################################



from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import re
import os
import torch
import faiss
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from sentence_transformers import SentenceTransformer
import vertexai
from vertexai.generative_models import GenerativeModel, Part



# =========================================================
# PDF TEXT PROCESSING & FAISS (COARSE RETRIEVAL)
# =========================================================
def chunk_text_with_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
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

def create_embeddings(model, chunks):
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  
    index.add(embeddings)
    return index

def search_indices(query, model, index, chunks, k=10):
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, k)
    return indices[0]   

def keyword_score(text, query):
    # 🔥 FIX: Ignore stop words and short words to prevent dense pages from hijacking the score
    stop_words = {"what", "is", "the", "of", "in", "on", "for", "with", "a", "an", "to", "as", "and", "by", "are"}
    query_words = [w for w in re.findall(r'\b\w+\b', query.lower()) if w not in stop_words and len(w) > 2]
    
    text = text.lower()
    # Score based on count of meaningful keywords
    return sum(text.count(word) for word in query_words)

def get_top_pages_hybrid(query, text_chunks, text_model, index, texts, k=5):
    # FAISS retrieval
    indices = search_indices(query, text_model, index, texts, k=k)
    faiss_pages = [text_chunks[i]["page"] for i in indices]

    # Keyword scoring
    page_scores = {}
    for chunk in text_chunks:
        page = chunk["page"]
        score = keyword_score(chunk["text"], query)
        if page not in page_scores:
            page_scores[page] = 0
        page_scores[page] += score

    keyword_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    keyword_pages = [p for p, _ in keyword_pages[:k]]

    final_pages = list(set(faiss_pages + keyword_pages))
    return final_pages


# =============================================================================
# =============================================================================
#
#   PHASE 6 — END-TO-END MULTIMODAL RAG (ColPali-like Approach)
#
#   PaliGemma 3B for embeddings  +  Gemini 2.5 Pro for generation
#   Overlapping patches  •  MaxSim late interaction  •  Hybrid reranking
#   Multi-page visual context  •  Enhanced source attribution
#
# =============================================================================
# =============================================================================

from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import time
import json

# =========================================================
# TASK 1 — VISUAL DOCUMENT INGESTION & PREPROCESSING
# =========================================================

def pdf_to_images_phase6(pdf_path, output_dir="page_images_phase6", scale=4):
    """
    Convert PDF pages to high-resolution images for Phase 6.
    Uses a higher scale factor (4x) than Phase 5 (3x) for better detail.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(path)

        image_paths.append({
            "path": path,
            "page": i + 1
        })

    print(f"[Phase 6] Converted {len(image_paths)} pages to images at {scale}x resolution")
    return image_paths


def create_overlapping_patches(image_path, page_number, patch_size=384, stride=192):
    """
    Create overlapping patches with configurable stride.
    Unlike Phase 5's non-overlapping 512x512 patches, overlapping patches
    ensure no content is missed at boundaries — critical for tables/charts
    that may span patch edges.
    """
    img = Image.open(image_path).convert("RGB")
    patches = []
    width, height = img.size

    for x in range(0, width, stride):
        for y in range(0, height, stride):
            # Ensure patch doesn't start beyond the image
            if x >= width or y >= height:
                continue

            patch = img.crop((
                x,
                y,
                min(x + patch_size, width),
                min(y + patch_size, height)
            ))

            # Skip very small edge patches (less than 25% of full patch area)
            pw, ph = patch.size
            if pw * ph < (patch_size * patch_size * 0.25):
                continue

            patches.append({
                "image": patch,
                "page": page_number,
                "coords": (x, y),
                "patch_size": (pw, ph),
                "source": image_path,
                "type": "patch"
            })

    return patches


def classify_patch_region(patch_image):
    """
    Simple heuristic to classify a patch as text, table, figure, or whitespace.
    Uses pixel variance and edge density analysis.
    """
    import numpy as np

    img_array = np.array(patch_image.convert("L"))  # grayscale

    # Metrics
    mean_val = img_array.mean()
    std_val = img_array.std()
    white_ratio = (img_array > 240).sum() / img_array.size

    # Classification heuristics
    if white_ratio > 0.95:
        return "whitespace"
    elif std_val > 60:
        return "figure"  # high contrast = likely chart/image
    elif std_val > 30:
        return "table"   # moderate contrast = likely table with lines
    else:
        return "text"    # low contrast = regular text


def create_smart_patches_phase6(image_path, page_number, patch_size=384, stride=192):
    """
    Create overlapping patches with region classification metadata.
    Filters out whitespace-only patches to reduce noise.
    """
    raw_patches = create_overlapping_patches(image_path, page_number, patch_size, stride)

    smart_patches = []
    for p in raw_patches:
        region_type = classify_patch_region(p["image"])

        # Skip pure whitespace patches
        if region_type == "whitespace":
            continue

        p["region_type"] = region_type
        smart_patches.append(p)

    return smart_patches


# =========================================================
# TASK 2 — MULTIMODAL EMBEDDING GENERATION
# =========================================================

def load_colpali_model():
    """
    Load PaliGemma 3B optimized for Phase 6 ColPali-style embeddings.
    Same base model as Phase 5 but with explicit memory management.
    """
    print("[Phase 6] Loading PaliGemma 3B for ColPali-style embeddings...")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto"
    )
    # Don't call model.to(device) — device_map="auto" already handles placement
    model.eval()

    # Infer actual device from model parameters
    device = str(next(model.parameters()).device)
    print(f"[Phase 6] PaliGemma loaded on {device}")
    return model, processor, device


def embed_patches_colpali(model, processor, patches, device, batch_size=1):
    """
    Generate token-level embeddings for patches using PaliGemma.
    Returns per-token embeddings for late-interaction (MaxSim) retrieval.

    Includes batch processing with memory management for large patch sets.
    """
    patch_token_embeddings = []
    total = len(patches)

    print(f"[Phase 6] Embedding {total} patches...")

    for i, p in enumerate(patches):
        if i % 25 == 0:
            print(f"  → Patch {i}/{total} ({100*i//total}%)")

        image = p["image"]
        prompt = "<image> describe the document"

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        token_embeddings = outputs.hidden_states[-1].squeeze(0)
        token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)
        patch_token_embeddings.append(token_embeddings.cpu().numpy())

        # Memory cleanup every 50 patches
        if i % 50 == 0 and device == "mps":
            torch.mps.empty_cache()
        elif i % 50 == 0 and device == "cuda":
            torch.cuda.empty_cache()

    print(f"[Phase 6] Embedding complete. {len(patch_token_embeddings)} patches embedded.")
    return patch_token_embeddings


def embed_query_colpali(model, processor, query, device):
    """
    Embed query through PaliGemma using a dummy image so query tokens
    pass through the same visual pathway as patch tokens.
    Returns token-level embeddings for MaxSim matching.
    """
    dummy_image = Image.new('RGB', (224, 224), (255, 255, 255))

    inputs = processor(images=dummy_image, text=query, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    token_embeddings = outputs.hidden_states[-1].squeeze(0)
    token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)

    return token_embeddings.cpu().numpy()


def create_qdrant_client_phase6():
    """Create in-memory Qdrant client for Phase 6."""
    client = QdrantClient(":memory:")
    return client


def store_embeddings_qdrant_phase6(client, collection_name, patches, patch_token_embeddings):
    """
    Store patch embeddings in Qdrant with full metadata.
    Since MaxSim uses token-level embeddings, we store the mean-pooled
    vector for initial filtering and keep token embeddings in memory.
    """
    # Mean-pool token embeddings for Qdrant storage
    mean_embeddings = []
    for tokens in patch_token_embeddings:
        mean_emb = tokens.mean(axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        mean_embeddings.append(mean_emb)

    vector_size = mean_embeddings[0].shape[0]

    # Create collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    # Insert points with metadata
    points = []
    for i, (patch, emb) in enumerate(zip(patches, mean_embeddings)):
        payload = {
            "page": patch["page"],
            "coords_x": patch["coords"][0],
            "coords_y": patch["coords"][1],
            "source": patch["source"],
            "region_type": patch.get("region_type", "unknown"),
            "patch_index": i
        }

        points.append(
            PointStruct(
                id=i,
                vector=emb.tolist(),
                payload=payload
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f"[Phase 6] Stored {len(points)} patch embeddings in Qdrant collection '{collection_name}'")

    return mean_embeddings


# =========================================================
# TASK 3 — MULTIMODAL RETRIEVAL (MaxSim / Late Interaction)
# =========================================================

def maxsim_score_phase6(query_tokens, patch_tokens):
    """
    Enhanced MaxSim scoring with softmax-weighted aggregation.
    For each query token, finds the best matching patch token,
    then aggregates using softmax weighting to emphasize strong matches.
    """
    # query_tokens → (Q, D), patch_tokens → (P, D)
    sim_matrix = np.matmul(query_tokens, patch_tokens.T)  # (Q, P)

    # For each query token → max similarity with any patch token
    max_sim_per_query = sim_matrix.max(axis=1)  # (Q,)

    # Softmax-weighted aggregation (emphasizes strong matches)
    exp_scores = np.exp(max_sim_per_query - max_sim_per_query.max())
    weights = exp_scores / exp_scores.sum()

    weighted_score = (weights * max_sim_per_query).sum()

    return weighted_score


def retrieve_patches_maxsim_phase6(query_tokens, patch_token_embeddings, patches, top_k=10):
    """
    Retrieve top-k patches using MaxSim late-interaction scoring.
    Returns patches sorted by relevance score.
    """
    scores = []

    for i, patch_tokens in enumerate(patch_token_embeddings):
        score = maxsim_score_phase6(query_tokens, patch_tokens)
        scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])

    results = []
    for score, idx in scores[:top_k]:
        patch = patches[idx].copy()
        patch["maxsim_score"] = float(score)
        results.append(patch)

    return results


def retrieve_pages_maxsim_phase6(query_tokens, patch_token_embeddings, patches, top_k_patches=15, top_k_pages=5):
    """
    Two-stage retrieval:
    1. Retrieve top-k patches via MaxSim
    2. Aggregate scores per page to identify best pages
    
    Returns (best_pages, retrieved_patches_with_scores)
    """
    # Stage 1: Get top patches
    retrieved_patches = retrieve_patches_maxsim_phase6(
        query_tokens, patch_token_embeddings, patches, top_k=top_k_patches
    )

    # Stage 2: Aggregate page scores
    page_scores = {}
    page_patch_counts = Counter()

    for p in retrieved_patches:
        page = p["page"]
        score = p["maxsim_score"]

        if page not in page_scores:
            page_scores[page] = 0.0

        page_scores[page] += score
        page_patch_counts[page] += 1

    # Normalize by patch count (average score per page)
    page_avg_scores = {
        page: page_scores[page] / page_patch_counts[page]
        for page in page_scores
    }

    # Sort by average score
    sorted_pages = sorted(page_avg_scores.items(), key=lambda x: x[1], reverse=True)
    best_pages = [page for page, _ in sorted_pages[:top_k_pages]]

    print(f"[Phase 6] MaxSim top pages: {best_pages}")
    return best_pages, retrieved_patches


def hybrid_rerank_phase6(query, text_chunks, text_model, faiss_index, texts,
                         query_tokens, patch_token_embeddings, patches,
                         k_text=15, k_patches=15, top_k_pages=5):
    """
    Hybrid reranking that fuses:
    1. Text-based coarse retrieval scores (FAISS + keyword)
    2. Visual MaxSim scores (PaliGemma patch embeddings)

    Returns the final ranked pages and the MaxSim retrieved patches.
    """
    # ── Text-based page scores ──
    text_top_pages = get_top_pages_hybrid(query, text_chunks, text_model, faiss_index, texts, k=k_text)

    # Score text pages by rank position
    text_page_scores = {}
    for rank, page in enumerate(text_top_pages):
        text_page_scores[page] = (len(text_top_pages) - rank)  # higher rank = higher score

    # ── Visual MaxSim page scores ──
    visual_pages, retrieved_patches = retrieve_pages_maxsim_phase6(
        query_tokens, patch_token_embeddings, patches,
        top_k_patches=k_patches, top_k_pages=top_k_pages * 2
    )

    visual_page_scores = {}
    for rank, page in enumerate(visual_pages):
        visual_page_scores[page] = (len(visual_pages) - rank)

    # ── Fuse scores ──
    all_pages = set(list(text_page_scores.keys()) + list(visual_page_scores.keys()))

    fused_scores = {}
    for page in all_pages:
        t_score = text_page_scores.get(page, 0)
        v_score = visual_page_scores.get(page, 0)

        # Weighted fusion: text retrieval is trusted more for coarse filtering
        fused_scores[page] = 0.6 * t_score + 0.4 * v_score

        # Bonus: keyword presence in page text (including acronyms like PCRF)
        page_text = " ".join([c["text"] for c in text_chunks if c["page"] == page]).lower()
        query_words = [w for w in re.findall(r'\b\w+\b', query.lower()) if len(w) > 2]
        keyword_hits = sum(1 for w in query_words if w in page_text)
        fused_scores[page] += keyword_hits * 1.0

    # Sort and select top pages
    sorted_pages = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    final_pages = [page for page, _ in sorted_pages[:top_k_pages]]

    print(f"[Phase 6] Hybrid reranked pages: {final_pages}")
    print(f"  Text pages: {text_top_pages[:top_k_pages]}")
    print(f"  Visual pages: {visual_pages[:top_k_pages]}")

    return final_pages, retrieved_patches


# =========================================================
# TASK 4 — CONTEXTUAL GENERATION WITH VISUAL CONTEXT
# =========================================================

def generate_answer_phase6(query, page_image_paths, text_context=""):
    """
    Multi-page visual generation using Gemini 2.5 Pro.
    Sends multiple retrieved page images (up to 3) along with
    extracted text context for cross-page synthesis.
    """
    model = GenerativeModel("gemini-2.5-pro")

    content_parts = []

    # Add page images (up to 3 for context window management)
    for i, img_path in enumerate(page_image_paths[:3]):
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        image_part = Part.from_data(
            data=image_bytes,
            mime_type="image/png"
        )
        content_parts.append(image_part)

    # Build prompt with text context if available
    text_section = ""
    if text_context:
        text_section = f"""
    SUPPLEMENTARY TEXT CONTEXT (extracted from the document):
    {text_context[:3000]}
    """

    prompt = f"""
    You are an elite financial document analyst. You are given {len(page_image_paths[:3])} page(s) from a financial report document.
    
    Your task is to answer the following question by carefully analyzing ALL provided pages.
    
    CRITICAL INSTRUCTIONS:
    1. CROSS-PAGE SYNTHESIS: The answer may span multiple pages. Check all pages thoroughly.
    2. TABLES & CHARTS: For tabular data, trace row labels to column headers precisely.
       Values in parentheses like "(30)" represent negative numbers.
    3. EXACT VALUES: Extract exact numerical values, percentages, and dates as written.
    4. CITE SOURCE: When possible, mention which page or section the answer comes from.
    5. DO NOT GUESS: If the information is clearly not in the provided pages, say so.
    
    {text_section}
    
    Question: {query}
    
    Provide a clear, precise answer.
    """

    content_parts.append(prompt)

    response = model.generate_content(content_parts)
    return response.text


# =========================================================
# TASK 5 — ENHANCED SOURCE ATTRIBUTION
# =========================================================

def draw_attribution_phase6(image_path, retrieved_patches, output_path="phase6_attribution.png", patch_size=384):
    """
    Enhanced source attribution with:
    - Color-coded boxes (red = top-1, orange = top-2/3, yellow = rest)
    - Score labels on each box
    - Semi-transparent overlay for highlight regions
    - Rank numbers for each attributed region
    """
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_main = ImageDraw.Draw(img)

    # Color scheme by rank
    colors = [
        (255, 0, 0, 80),      # Rank 1: Red (semi-transparent fill)
        (255, 140, 0, 60),     # Rank 2: Orange
        (255, 200, 0, 50),     # Rank 3: Yellow
        (0, 200, 0, 40),       # Rank 4+: Green
    ]

    border_colors = [
        (255, 0, 0),           # Red border
        (255, 140, 0),         # Orange border
        (255, 200, 0),         # Yellow border
        (0, 200, 0),           # Green border
    ]

    matching_patches = [p for p in retrieved_patches if p.get("source") == image_path]

    for rank, p in enumerate(matching_patches):
        x, y = p["coords"]
        pw, ph = p.get("patch_size", (patch_size, patch_size))
        score = p.get("maxsim_score", 0)

        color_idx = min(rank, len(colors) - 1)

        # Semi-transparent fill
        draw_overlay.rectangle(
            [x, y, x + pw, y + ph],
            fill=colors[color_idx]
        )

        # Solid border
        draw_main.rectangle(
            [x, y, x + pw, y + ph],
            outline=border_colors[color_idx],
            width=4
        )

        # Rank + score label
        label = f"#{rank+1} ({score:.3f})"
        draw_main.text((x + 5, y + 5), label, fill=border_colors[color_idx])

    # Composite
    result = Image.alpha_composite(img, overlay).convert("RGB")

    # Add legend at bottom
    legend_height = 40
    legend_img = Image.new("RGB", (result.width, result.height + legend_height), (255, 255, 255))
    legend_img.paste(result, (0, 0))
    draw_legend = ImageDraw.Draw(legend_img)

    legend_items = [
        ("■ Rank 1 (Highest)", (255, 0, 0)),
        ("■ Rank 2-3", (255, 140, 0)),
        ("■ Rank 4+", (0, 200, 0))
    ]

    x_offset = 10
    for text, color in legend_items:
        draw_legend.text((x_offset, result.height + 10), text, fill=color)
        x_offset += 200

    legend_img.save(output_path)
    return output_path


def create_attribution_report(query, answer, retrieved_patches, page_images, output_dir="phase6_attributions"):
    """
    Generate a per-page attribution report with annotated images.
    Returns paths to all generated attribution images.
    """
    os.makedirs(output_dir, exist_ok=True)
    attribution_paths = []

    # Group patches by source image
    patches_by_source = {}
    for p in retrieved_patches:
        source = p.get("source", "")
        if source not in patches_by_source:
            patches_by_source[source] = []
        patches_by_source[source].append(p)

    # Generate attribution for each source page
    for source_path, source_patches in patches_by_source.items():
        page_num = source_patches[0]["page"]
        output_path = os.path.join(output_dir, f"attribution_page_{page_num}.png")

        draw_attribution_phase6(
            source_path,
            source_patches,
            output_path=output_path
        )
        attribution_paths.append({
            "path": output_path,
            "page": page_num,
            "num_patches": len(source_patches)
        })

    # Save metadata
    report = {
        "query": query,
        "answer": answer[:500],
        "attributed_pages": [a["page"] for a in attribution_paths],
        "total_patches": len(retrieved_patches)
    }

    report_path = os.path.join(output_dir, "attribution_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[Phase 6] Attribution report saved to {output_dir}/")
    return attribution_paths


# =========================================================
# FULL PHASE 6 PIPELINE WRAPPER
# =========================================================

def run_phase6_pipeline(query, pdf_path="data/ifc_report.pdf", top_k_pages=3, max_patches=400):
    """
    End-to-end Phase 6 pipeline:
    1. Coarse text retrieval → page filtering
    2. Visual ingestion → overlapping smart patches
    3. PaliGemma embedding (patches + query)
    4. Hybrid reranking (text + visual MaxSim)
    5. Multi-page Gemini 2.5 Pro generation
    6. Enhanced source attribution
    
    Returns dict with answer, attribution paths, timing, and metadata.
    """
    timings = {}
    t_start = time.time()

    # ── STEP 1: Coarse text retrieval ──
    t0 = time.time()
    print(f"\n[Phase 6] Processing query: '{query}'")
    print("[Phase 6 Step 1] Running coarse text retrieval...")

    text_chunks = chunk_text_with_metadata(pdf_path)
    texts = [c["text"] for c in text_chunks]

    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_embeddings = create_embeddings(text_model, texts)
    faiss_index = create_faiss_index(text_embeddings)

    coarse_pages = get_top_pages_hybrid(query, text_chunks, text_model, faiss_index, texts, k=20)
    timings["text_retrieval"] = time.time() - t0
    print(f"  Candidate pages: {coarse_pages}")

    # ── STEP 2: Visual ingestion ──
    t0 = time.time()
    print("[Phase 6 Step 2] Converting pages to visual patches...")

    images = pdf_to_images_phase6(pdf_path)
    filtered_images = [img for img in images if img["page"] in coarse_pages]

    patches_by_page = {}
    for img in filtered_images:
        page_patches = create_smart_patches_phase6(img["path"], img["page"])
        patches_by_page[img["page"]] = page_patches

    total_patches = sum(len(p) for p in patches_by_page.values())
    print(f"  Generated {total_patches} smart patches (filtered whitespace)")

    # Distribute patch budget evenly across pages so every candidate page
    # gets visual coverage (the old code just took the first N in doc order,
    # which starved later pages like page 8)
    if total_patches > max_patches and len(patches_by_page) > 0:
        per_page_budget = max(max_patches // len(patches_by_page), 4)
        print(f"  Limiting to {max_patches} patches ({per_page_budget} per page across {len(patches_by_page)} pages)")
        patches = []
        for page in sorted(patches_by_page.keys()):
            page_p = patches_by_page[page]
            # Take evenly spaced patches if over budget for this page
            if len(page_p) > per_page_budget:
                step = len(page_p) / per_page_budget
                page_p = [page_p[int(i * step)] for i in range(per_page_budget)]
            patches.extend(page_p)
        # Final trim if rounding pushed us over
        patches = patches[:max_patches]
    else:
        patches = []
        for page in sorted(patches_by_page.keys()):
            patches.extend(patches_by_page[page])

    timings["visual_ingestion"] = time.time() - t0

    # ── STEP 3: PaliGemma embeddings ──
    t0 = time.time()
    print("[Phase 6 Step 3] Generating PaliGemma embeddings...")

    pali_model, pali_processor, device = load_colpali_model()

    patch_token_embeddings = embed_patches_colpali(
        pali_model, pali_processor, patches, device
    )

    query_tokens = embed_query_colpali(pali_model, pali_processor, query, device)
    timings["embedding"] = time.time() - t0

    # ── STEP 4: Hybrid reranking ──
    t0 = time.time()
    print("[Phase 6 Step 4] Running hybrid reranking...")

    final_pages, retrieved_patches = hybrid_rerank_phase6(
        query, text_chunks, text_model, faiss_index, texts,
        query_tokens, patch_token_embeddings, patches,
        k_text=20, k_patches=15, top_k_pages=top_k_pages
    )
    timings["retrieval"] = time.time() - t0

    # ── STEP 5: Multi-page generation ──
    t0 = time.time()
    print(f"[Phase 6 Step 5] Generating answer from pages {final_pages}...")

    # Get image paths for final pages
    page_image_paths = [
        img["path"] for img in filtered_images
        if img["page"] in final_pages
    ]

    # Also try to include pages from all images if not in filtered set
    for page in final_pages:
        matching = [img["path"] for img in images if img["page"] == page]
        if matching and matching[0] not in page_image_paths:
            page_image_paths.append(matching[0])

    # Build supplementary text context from top pages
    text_context = "\n\n".join([
        c["text"] for c in text_chunks
        if c["page"] in final_pages
    ])

    answer = generate_answer_phase6(query, page_image_paths, text_context)
    timings["generation"] = time.time() - t0

    # ── STEP 6: Source attribution ──
    t0 = time.time()
    print("[Phase 6 Step 6] Generating source attribution...")

    attribution_paths = create_attribution_report(
        query, answer, retrieved_patches, filtered_images
    )
    timings["attribution"] = time.time() - t0

    timings["total"] = time.time() - t_start

    # ── Store in Qdrant for persistence ──
    try:
        qdrant_client = create_qdrant_client_phase6()
        store_embeddings_qdrant_phase6(
            qdrant_client, "phase6_patches",
            patches, patch_token_embeddings
        )
    except Exception as e:
        print(f"[Phase 6] Qdrant storage skipped: {e}")

    result = {
        "answer": answer,
        "final_pages": final_pages,
        "retrieved_patches": retrieved_patches,
        "attribution_paths": attribution_paths,
        "timings": timings,
        "num_patches_processed": len(patches),
        "page_image_paths": page_image_paths
    }

    print(f"\n{'='*60}")
    print(f"[Phase 6] COMPLETE — Answer from pages {final_pages}")
    print(f"  Total time: {timings['total']:.1f}s")
    print(f"  Patches processed: {len(patches)}")
    print(f"{'='*60}")

    return result


# =========================================================
# TASK 6 — PIPELINE COMPARISON (Phase 5 vs Phase 6)
# =========================================================

def run_phase5_pipeline(query, pdf_path="data/ifc_report.pdf"):
    """
    Run the existing Phase 5 pipeline and return results in a comparable format.
    Wraps the Phase 5 functions already defined in this file.
    """
    t_start = time.time()

    # Text retrieval
    text_chunks = chunk_text_with_metadata(pdf_path)
    texts = [c["text"] for c in text_chunks]

    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_embeddings = create_embeddings(text_model, texts)
    index = create_faiss_index(text_embeddings)

    top_pages = get_top_pages_hybrid(query, text_chunks, text_model, index, texts, k=25)

    # Visual pipeline
    images = pdf_to_images(pdf_path)
    filtered_images = [img for img in images if img["page"] in top_pages]

    patches = []
    for img in filtered_images:
        patches.extend(create_image_patches(img["path"], img["page"]))

    patches_small = patches[:400]

    # PaliGemma embeddings
    model, processor, device = load_paligemma_model()

    patch_token_embeddings = []
    for p in patches_small:
        tokens = embed_patch_tokens_paligemma(model, processor, p, device)
        patch_token_embeddings.append(tokens)

    query_tokens = embed_query_tokens_paligemma(model, processor, query, device)

    # MaxSim retrieval
    retrieved_patches = retrieve_patches_maxsim_paligemma(
        query_tokens, patch_token_embeddings, patches_small, top_k=3
    )

    # Generation
    best_patch = retrieved_patches[0]
    best_page = best_patch["page"]
    best_page_image = [img["path"] for img in filtered_images if img["page"] == best_page][0]

    answer = generate_answer_multimodal_gemini(query, best_page_image)

    total_time = time.time() - t_start

    return {
        "answer": answer,
        "final_pages": [best_page],
        "total_time": total_time,
        "num_patches": len(patches_small)
    }


def compare_pipelines(queries, pdf_path="data/ifc_report.pdf"):
    """
    Run Phase 5 and Phase 6 on the same queries and compare results.
    Returns a comparison dataframe.
    """
    import pandas as pd

    comparison = []

    for i, query in enumerate(queries):
        print(f"\n{'='*60}")
        print(f"QUERY {i+1}/{len(queries)}: {query}")
        print(f"{'='*60}")

        # Phase 5
        print("\n── Running Phase 5 ──")
        try:
            p5_result = run_phase5_pipeline(query, pdf_path)
            p5_answer = p5_result["answer"]
            p5_pages = p5_result["final_pages"]
            p5_time = p5_result["total_time"]
        except Exception as e:
            print(f"Phase 5 failed: {e}")
            p5_answer = f"ERROR: {e}"
            p5_pages = []
            p5_time = 0

        # Phase 6
        print("\n── Running Phase 6 ──")
        try:
            p6_result = run_phase6_pipeline(query, pdf_path)
            p6_answer = p6_result["answer"]
            p6_pages = p6_result["final_pages"]
            p6_time = p6_result["timings"]["total"]
        except Exception as e:
            print(f"Phase 6 failed: {e}")
            p6_answer = f"ERROR: {e}"
            p6_pages = []
            p6_time = 0

        comparison.append({
            "query": query,
            "phase5_answer": p5_answer,
            "phase5_pages": str(p5_pages),
            "phase5_time_s": round(p5_time, 1),
            "phase6_answer": p6_answer,
            "phase6_pages": str(p6_pages),
            "phase6_time_s": round(p6_time, 1)
        })

    df = pd.DataFrame(comparison)
    return df
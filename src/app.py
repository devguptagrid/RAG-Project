import streamlit as st
from rag import *
from sentence_transformers import SentenceTransformer
import vertexai
from dotenv import load_dotenv
from langfuse import Langfuse, observe
load_dotenv()



langfuse = Langfuse()

@st.cache_resource
def load_pipeline():
    pdf_path = "data/ifc_report.pdf"
    
    text = extract_text_from_pdf(pdf_path)
    chunks = filter_chunks(chunk_text(text))
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = create_embeddings(model, chunks)
    
    index = create_faiss_index(embeddings)
    
    vertexai.init(project="gd-gcp-gridu-genai", location="us-central1")
    
    return model, index, chunks


model, index, chunks = load_pipeline()

@observe(name="rag-query")
def run_rag_pipeline(query):
    
    retrieved = search(query, model, index, chunks, k=10)
    results = cross_rerank(query, retrieved, k=5)
    
    answer = generate_answer(query, results)
    
    return {
        "answer": answer,
        "retrieved_chunks": results
    }

load_dotenv()


st.title("📄 IFC RAG System")

query = st.text_input("Ask a question about the IFC report:")

if query:
    with st.spinner("Thinking..."):
        answer = run_rag_pipeline(query)
        st.subheader("Answer:")
        st.success(answer['answer'])

    with st.expander("🔍 Retrieved Context"):
        for i, chunk in enumerate(answer["retrieved_chunks"]):
            st.write(f"Chunk {i+1}:")
            st.write(chunk)
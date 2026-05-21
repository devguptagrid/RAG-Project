
import streamlit as st
from rag import *
from sentence_transformers import SentenceTransformer
import vertexai
from dotenv import load_dotenv
from langfuse import Langfuse, observe
load_dotenv()
langfuse = Langfuse()

st.sidebar.title("RAG Dashboard")

phase = st.sidebar.selectbox(
    "Select Phase",
    ["Phase 1: Basic RAG application", "Phase 2: RAG Evaluation", "Phase 3: Hybrid Search+Metadata", "Phase 4: Multimodel RAG", "Phase 5: Multimodel RAG with Colpali-like approach"]
)

@st.cache_resource
def load_phase1_pipeline():
    pdf_path = "data/ifc_report.pdf"
    
    text = extract_text_from_pdf(pdf_path)
    chunks = filter_chunks(chunk_text(text))
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = create_embeddings(model, chunks)
    
    index = create_faiss_index(embeddings)
    
    vertexai.init(project="gd-gcp-gridu-genai", location="us-central1")
    
    return model, index, chunks,embeddings


@st.cache_resource
def load_phase3_pipeline():
    pdf_path = "data/ifc_report.pdf"

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 🔥 metadata chunks
    metadata_chunks = chunk_text_with_metadata(pdf_path)
    texts = [c["text"] for c in metadata_chunks]

    embeddings = create_embeddings(model, texts)
    index_meta = create_faiss_index(embeddings)

    return model, index_meta, metadata_chunks,embeddings





@observe(name="rag-query")
def run_rag_pipeline(query):
    
    retrieved = search_text(query, model1, index1, chunks, k=10)
    results = cross_rerank(query, retrieved, k=5)
    
    client = create_qdrant_client()

    collection_name = "ifc_collection"

    create_collection(client, collection_name, embeddings1.shape[1])

    insert_into_qdrant(client, collection_name, chunks, embeddings1)



    retrieved = search_qdrant(client, collection_name, query, model1, k=10)
    results = cross_rerank(query, retrieved,k=5)
    answer = generate_answer(query, results)
    
    return {
        "answer": answer,
        "retrieved_chunks": results
    }

if phase == "Phase 1: Basic RAG application":
    st.header("Phase 1: Basic RAG Application")
    model1, index1, chunks,embeddings1 = load_phase1_pipeline()

    # 🔹 Select vector store
    vector_store = st.selectbox(
        "Select Vector Store",
        ["FAISS", "Qdrant"]
    )

    # 🔹 Query input
    query = st.text_input("Ask a question about the IFC report:")

    if query:
        with st.spinner("Thinking..."):

            # Setup Qdrant only if needed
            if vector_store == "Qdrant":
                client = create_qdrant_client()
                collection_name = "ifc_collection"

                create_collection(client, collection_name, embeddings1.shape[1])
                insert_into_qdrant(client, collection_name, chunks, embeddings1)

            # 🔹 Retrieval
            if vector_store == "FAISS":
                retrieved = search_text(query, model1, index1, chunks, k=10)
            elif vector_store == "Qdrant":
                retrieved = search_qdrant(client, collection_name, query, model1, k=10)

            # 🔹 Generate answer
            answer = generate_answer(query, retrieved)

            # 🔹 Show answer
            st.subheader("Answer:")
            st.success(answer)

            # 🔹 Show retrieved chunks
            with st.expander("🔍 Retrieved Context"):
                for i, chunk in enumerate(retrieved):
                    st.write(f"Chunk {i+1}:")
                    st.write(chunk)

elif phase == "Phase 2: RAG Evaluation":
    st.header("Phase 2: RAG Evaluation")

    # 🔹 Scores (hardcoded for now)
    faiss_scores = {
        "faithfulness": 0.9706,
        "answer_relevancy": 0.4580,
        "context_precision": 0.5457,
        "context_recall": 0.7206
    }

    qdrant_scores = {
        "faithfulness": 0.9228,
        "answer_relevancy": 0.4358,
        "context_precision": 0.5611,
        "context_recall": 0.7353
    }

    # 🔹 Display metrics side by side
    st.subheader("Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### FAISS")
        for k, v in faiss_scores.items():
            st.metric(label=k, value=round(v, 4))

    with col2:
        st.markdown("### Qdrant")
        for k, v in qdrant_scores.items():
            st.metric(label=k, value=round(v, 4))


    # 🔹 Comparison
    st.subheader("Comparison")

    comparison_text = []

    for metric in faiss_scores:
        if faiss_scores[metric] > qdrant_scores[metric]:
            comparison_text.append(f"✔ FAISS performs better in **{metric}**")
        elif faiss_scores[metric] < qdrant_scores[metric]:
            comparison_text.append(f"✔ Qdrant performs better in **{metric}**")
        else:
            comparison_text.append(f"➖ Both perform equally in **{metric}**")

    for line in comparison_text:
        st.write(line)


    

elif phase == "Phase 3: Hybrid Search+Metadata":
    
    st.header("🔍 Phase 3: Hybrid Search + Metadata Filtering")
    model3, index_meta, metadata_chunks ,embeddings2= load_phase3_pipeline()
    # 🔹 Query input
    query = st.text_input("Ask a question about the document:")

    # 🔹 Checkbox for full document
    use_full_doc = st.checkbox("Search entire document", value=True)

    # 🔹 Page range inputs
    if not use_full_doc:
        col1, col2 = st.columns(2)

        with col1:
            start_page = st.number_input("Start Page", min_value=1, value=1)

        with col2:
            end_page = st.number_input("End Page", min_value=1, value=10)
    else:
        start_page, end_page = None, None

    # 🔹 Run pipeline
    if query:
        with st.spinner("Running advanced RAG pipeline..."):

            result = run_phase3_pipeline(
                query=query,
                model=model3,
                index=index_meta,
                metadata_chunks=metadata_chunks,
                start_page=start_page,
                end_page=end_page
            )

            # 🔹 Show answer
            st.subheader("📌 Answer")
            st.success(result["answer"])

            # 🔹 Show retrieved chunks
            with st.expander("🔍 Retrieved Context (with page numbers)"):
                for i, chunk in enumerate(result["retrieved_chunks"]):
                    st.markdown(f"**Chunk {i+1} (Page {chunk['page']})**")
                    st.write(chunk["text"])
                    st.markdown("---")

elif phase == "Phase 4: Multimodel RAG":

    st.header("🧠 Phase 4: Multimodal Retrieval (Text + Images)")

    # 🔹 Load data
    @st.cache_resource
    def load_phase4_pipeline():
        pdf_path = "data/ifc_report.pdf"

        model = SentenceTransformer('all-MiniLM-L6-v2')

        # TEXT
        text_chunks = chunk_text_with_metadata(pdf_path)

        # IMAGES
        images = extract_images(pdf_path)
        image_chunks = []

        for img in images:
            if img["path"].endswith(".jpx"):
                continue

            try:
                desc = describe_image(img["path"])

                if is_useful_image(desc):
                    image_chunks.append({
                        "text": desc,
                        "type": "image",
                        "page": img["page"]
                    })

            except:
                pass

        # 🔥 MERGE TEXT + IMAGES
        all_chunks = text_chunks + image_chunks

        texts = [c["text"] for c in all_chunks]

        embeddings = create_embeddings(model, texts)
        index = create_faiss_index(embeddings)

        return model, index, all_chunks

    model4, index4, all_chunks4 = load_phase4_pipeline()

    # 🔹 Query
    query = st.text_input("Ask a multimodal question (text + charts):")

    if query:
        with st.spinner("Thinking across text and visuals..."):

            texts = [c["text"] for c in all_chunks4]

            indices = search_indices(query, model4, index4, texts, k=10)
            retrieved = [all_chunks4[i] for i in indices]

            # 🔥 Simple multimodal reasoning
            context_chunks = [c["text"] for c in retrieved]

            answer = generate_answer(query, context_chunks)

        # 🔹 Output
        st.subheader("📌 Answer")
        st.success(answer)

        # 🔹 Show retrieved context
        with st.expander("🔍 Retrieved Context"):
            for i, c in enumerate(retrieved):
                st.markdown(f"**Chunk {i+1} | Page {c['page']} | Type: {c['type']}**")
                st.write(c["text"][:500])
                st.markdown("---")

elif phase == "Phase 5: Multimodel RAG with Colpali-like approach":
    st.info("Coming soon")





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
                st.markdown(f"**Chunk {i+1} | Page {c['page']}**")
                st.write(c["text"][:500])
                st.markdown("---")

elif phase == "Phase 5: Multimodel RAG with Colpali-like approach":

    st.header("🧠 Phase 5: Multimodal RAG with ColPali-like Approach")
    st.markdown("""
    **End-to-end multimodal RAG pipeline** using:
    - **PaliGemma 3B** for visual patch embeddings (ColPali-style)
    - **MaxSim / Late Interaction** retrieval
    - **Gemini 2.5 Pro** for multimodal answer generation
    - **Enhanced source attribution** with color-coded bounding boxes
    """)

    # ── CACHED HELPERS FOR STREAMLIT SPEED ──
    @st.cache_resource
    def get_cached_pali_model():
        return load_colpali_model()

    @st.cache_resource
    def get_cached_text_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    @st.cache_data
    def get_cached_text_chunks(pdf_path):
        return chunk_text_with_metadata(pdf_path)

    @st.cache_data
    def get_cached_pdf_images(pdf_path):
        return pdf_to_images_phase6(pdf_path)

    # 🔹 Query input
    query = st.text_input("Ask a question about the IFC report:", key="phase5_query")

    # 🔹 Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            top_k_pages = st.slider("Top-K Pages", min_value=1, max_value=5, value=3)
        with col2:
            max_patches = st.slider("Max Patches", min_value=100, max_value=800, value=400, step=50)
        with col3:
            patch_size = st.selectbox("Patch Size", [256, 384, 512], index=1)

    if query:
        pdf_path = "data/ifc_report.pdf"

        # Progress tracking
        progress_bar = st.progress(0, text="Initializing pipeline...")

        try:
            # ── STEP 1: Coarse text retrieval ──
            progress_bar.progress(5, text="Step 1/6: Running coarse text retrieval...")

            text_chunks = get_cached_text_chunks(pdf_path)
            texts = [c["text"] for c in text_chunks]

            text_model_st = get_cached_text_model()
            text_embeddings = create_embeddings(text_model_st, texts)
            faiss_index = create_faiss_index(text_embeddings)

            coarse_pages = get_top_pages_hybrid(query, text_chunks, text_model_st, faiss_index, texts, k=20)

            progress_bar.progress(15, text="Step 2/6: Converting pages to visual patches...")

            # ── STEP 2: Visual ingestion ──
            images = get_cached_pdf_images(pdf_path)
            filtered_images = [img for img in images if img["page"] in coarse_pages]

            patches_by_page = {}
            for img in filtered_images:
                page_patches = create_smart_patches_phase6(img["path"], img["page"], patch_size=patch_size, stride=patch_size // 2)
                patches_by_page[img["page"]] = page_patches

            total_patches = sum(len(p) for p in patches_by_page.values())

            # Distribute patch budget evenly across pages
            if total_patches > max_patches and len(patches_by_page) > 0:
                per_page_budget = max(max_patches // len(patches_by_page), 4)
                patches = []
                for page in sorted(patches_by_page.keys()):
                    page_p = patches_by_page[page]
                    if len(page_p) > per_page_budget:
                        step = len(page_p) / per_page_budget
                        page_p = [page_p[int(i * step)] for i in range(per_page_budget)]
                    patches.extend(page_p)
                patches = patches[:max_patches]
            else:
                patches = []
                for page in sorted(patches_by_page.keys()):
                    patches.extend(patches_by_page[page])

            progress_bar.progress(25, text=f"Step 3/6: Generating PaliGemma embeddings for {len(patches)} patches...")

            # ── STEP 3: PaliGemma embeddings ──
            pali_model, pali_processor, device = get_cached_pali_model()

            patch_token_embeddings = embed_patches_colpali(
                pali_model, pali_processor, patches, device
            )

            query_tokens = embed_query_colpali(pali_model, pali_processor, query, device)

            progress_bar.progress(70, text="Step 4/6: Running hybrid reranking...")

            # ── STEP 4: Hybrid reranking ──
            final_pages, retrieved_patches = hybrid_rerank_phase6(
                query, text_chunks, text_model_st, faiss_index, texts,
                query_tokens, patch_token_embeddings, patches,
                k_text=20, k_patches=15, top_k_pages=top_k_pages
            )

            progress_bar.progress(80, text=f"Step 5/6: Generating answer from pages {final_pages}...")

            # ── STEP 5: Multi-page generation ──
            page_image_paths = [
                img["path"] for img in filtered_images
                if img["page"] in final_pages
            ]

            for page in final_pages:
                matching = [img["path"] for img in images if img["page"] == page]
                if matching and matching[0] not in page_image_paths:
                    page_image_paths.append(matching[0])

            text_context = "\n\n".join([
                c["text"] for c in text_chunks
                if c["page"] in final_pages
            ])

            answer = generate_answer_phase6(query, page_image_paths, text_context)

            progress_bar.progress(90, text="Step 6/6: Generating source attribution...")

            # ── STEP 6: Source attribution ──
            attribution_paths = create_attribution_report(
                query, answer, retrieved_patches, filtered_images
            )

            progress_bar.progress(100, text="✅ Complete!")

            # =========================================================
            # DISPLAY RESULTS
            # =========================================================

            # Answer
            st.subheader("📌 Answer")
            st.success(answer)

            # Retrieved pages
            st.subheader("📄 Retrieved Pages")
            page_cols = st.columns(min(len(final_pages), 3))
            for i, page in enumerate(final_pages):
                with page_cols[i % 3]:
                    st.metric(f"Page {page}", f"Rank #{i+1}")

            # Source Attribution
            st.subheader("🎯 Source Attribution")
            if attribution_paths:
                attr_cols = st.columns(min(len(attribution_paths), 2))
                for i, attr in enumerate(attribution_paths):
                    with attr_cols[i % 2]:
                        st.markdown(f"**Page {attr['page']}** ({attr['num_patches']} relevant patches)")
                        st.image(attr["path"], caption=f"Attribution — Page {attr['page']}")

            # Retrieved Patches Detail
            with st.expander("🔍 Retrieved Patches (Top 10)"):
                for i, p in enumerate(retrieved_patches[:10]):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(p["image"], caption=f"Patch {i+1}", width=150)
                    with col2:
                        st.markdown(f"""
                        **Rank:** #{i+1}  
                        **Page:** {p['page']}  
                        **Coords:** ({p['coords'][0]}, {p['coords'][1]})  
                        **Region Type:** {p.get('region_type', 'N/A')}  
                        **MaxSim Score:** {p.get('maxsim_score', 'N/A'):.4f}
                        """)
                    st.markdown("---")

            # Text Context
            with st.expander("📝 Supplementary Text Context"):
                for page in final_pages:
                    st.markdown(f"**Page {page}:**")
                    page_text = "\n".join([c["text"] for c in text_chunks if c["page"] == page])
                    st.text(page_text[:1000])
                    st.markdown("---")

            # Pipeline Stats
            with st.expander("📊 Pipeline Statistics"):
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Candidate Pages", len(coarse_pages))
                with stat_cols[1]:
                    st.metric("Total Patches", len(patches))
                with stat_cols[2]:
                    st.metric("Final Pages", len(final_pages))
                with stat_cols[3]:
                    st.metric("Top Patches", len(retrieved_patches))

                # Region type breakdown
                region_counts = {}
                for p in patches:
                    rt = p.get("region_type", "unknown")
                    region_counts[rt] = region_counts.get(rt, 0) + 1

                st.markdown("**Patch Region Types:**")
                for rt, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"  • {rt}: {count}")

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

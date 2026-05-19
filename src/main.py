from rag import *
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":

    pdf_path = "data/ifc_report.pdf"

    # 🔹 Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # =========================================================
    # 🔥 PHASE 1: FAISS (Baseline)
    # =========================================================

    text = extract_text_from_pdf(pdf_path)
    chunks = filter_chunks(chunk_text(text))

    print("Total chunks:", len(chunks))

    embeddings = create_embeddings(model, chunks)
    index_faiss = create_faiss_index(embeddings)

    query = "What is the mission of IFC?"

    retrieved = search_text(query, model, index_faiss, chunks, k=10)

    answer = generate_answer(query, retrieved)

    print("\n🔹 FAISS Answer:\n", answer)


    # =========================================================
    # 🔥 PHASE 1: QDRANT (Baseline)
    # =========================================================

    client = create_qdrant_client()
    collection_name = "ifc_collection"

    create_collection(client, collection_name, embeddings.shape[1])
    insert_into_qdrant(client, collection_name, chunks, embeddings)

    retrieved_qdrant = search_qdrant(client, collection_name, query, model, k=10)

    print("\n🔹 Qdrant Results:\n")

    for i, res in enumerate(retrieved_qdrant):
        print(f"\nResult {i+1}:\n", res)

    answer_qdrant = generate_answer(query, retrieved_qdrant)

    print("\n🔹 Qdrant Answer:\n", answer_qdrant)


    # =========================================================
    # 🔥 PHASE 3: Hybrid (Metadata + Rerank)
    # =========================================================

    metadata_chunks = chunk_text_with_metadata(pdf_path)

    texts = [c["text"] for c in metadata_chunks]

    embeddings_meta = create_embeddings(model, texts)
    index_meta = create_faiss_index(embeddings_meta)

    result = run_phase3_pipeline(
        query="What is the official mission statement of IFC?",
        model=model,
        index=index_meta,
        metadata_chunks=metadata_chunks,
        start_page=None,
        end_page=None
    )

    print("\n🔹 Phase 3 Answer:\n", result["answer"])
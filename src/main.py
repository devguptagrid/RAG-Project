from rag import *
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":

    pdf_path = "data/ifc_report.pdf"

    # 🔹 Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # # =========================================================
    # # 🔥 PHASE 1: FAISS (Baseline)
    # # =========================================================

    # text = extract_text_from_pdf(pdf_path)
    # chunks = filter_chunks(chunk_text(text))

    # print("Total chunks:", len(chunks))

    # embeddings = create_embeddings(model, chunks)
    # index_faiss = create_faiss_index(embeddings)

    # query = "What is the mission of IFC?"

    # retrieved = search_text(query, model, index_faiss, chunks, k=10)

    # answer = generate_answer(query, retrieved)

    # print("\n🔹 FAISS Answer:\n", answer)


    # # =========================================================
    # # 🔥 PHASE 1: QDRANT (Baseline)
    # # =========================================================

    # client = create_qdrant_client()
    # collection_name = "ifc_collection"

    # create_collection(client, collection_name, embeddings.shape[1])
    # insert_into_qdrant(client, collection_name, chunks, embeddings)

    # retrieved_qdrant = search_qdrant(client, collection_name, query, model, k=10)

    # print("\n🔹 Qdrant Results:\n")

    # for i, res in enumerate(retrieved_qdrant):
    #     print(f"\nResult {i+1}:\n", res)

    # answer_qdrant = generate_answer(query, retrieved_qdrant)

    # print("\n🔹 Qdrant Answer:\n", answer_qdrant)


    # # =========================================================
    # # 🔥 PHASE 3: Hybrid (Metadata + Rerank)
    # # =========================================================

    # metadata_chunks = chunk_text_with_metadata(pdf_path)

    # texts = [c["text"] for c in metadata_chunks]

    # embeddings_meta = create_embeddings(model, texts)
    # index_meta = create_faiss_index(embeddings_meta)

    # result = run_phase3_pipeline(
    #     query="What is the official mission statement of IFC?",
    #     model=model,
    #     index=index_meta,
    #     metadata_chunks=metadata_chunks,
    #     start_page=None,
    #     end_page=None
    # )

    # print("\n🔹 Phase 3 Answer:\n", result["answer"])


    # =========================================================
    # 🔥 STEP 1: TEXT CHUNKS
    # =========================================================
    text_chunks = chunk_text_with_metadata(pdf_path)

    # =========================================================
    # 🔥 STEP 2: TABLE CHUNKS
    # =========================================================
    tables = extract_tables(pdf_path)
    table_chunks = create_table_chunks(tables)

    # =========================================================
    # 🔥 STEP 3: IMAGE CHUNKS (FIXED POSITION)
    # =========================================================
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

        except Exception as e:
            print(f"Skipping {img['path']} → {e}")

    print("Total images:", len(images))
    print("Useful images:", len(image_chunks))

    # =========================================================
    # 🔥 STEP 4: FIGURE CHUNKS (IMPORTANT ADDITION)
    # =========================================================
    figure_chunks = []

    for c in text_chunks:
        text = c["text"].lower()

        if "figure" in text or "chart" in text:
            figure_chunks.append({
                "text": c["text"],
                "type": "image",
                "page": c["page"]
            })

    # =========================================================
    # 🔥 STEP 5: MERGE EVERYTHING
    # =========================================================
    all_chunks = text_chunks + table_chunks + image_chunks + figure_chunks

    print("Total chunks:", len(all_chunks))

    # =========================================================
    # 🔥 STEP 6: EMBEDDINGS + INDEX
    # =========================================================
    texts = [c["text"] for c in all_chunks]

    embeddings = create_embeddings(model, texts)
    index = create_faiss_index(embeddings)

    # =========================================================
    # 🔥 STEP 7: QUERY
    # =========================================================
    query = "What was the total value of IFC's assets as of June 30, 2023?"

    indices = search_indices(query, model, index, texts, k=5)

    retrieved = [all_chunks[i] for i in indices]

    # =========================================================
    # 🔥 STEP 8: TABLE BOOSTING (FINAL CORRECT VERSION)
    # =========================================================

    if any(word in query.lower() for word in ["table", "values", "data", "numbers", "fy", "year", "income", "variance", "percentage", "%", "share", "accounted"]):

        # 🔹 split FAISS results
        retrieved_tables = [c for c in retrieved if c["type"] == "table"]
        retrieved_texts = [c for c in retrieved if c["type"] == "text"]

        # 🔥 If FAISS didn't return tables → fallback
        if len(retrieved_tables) == 0:

            # take nearby chunks (same pages as retrieved text)
            pages = set([c["page"] for c in retrieved_texts])

            fallback_tables = [
                c for c in table_chunks
                if c["page"] in pages
            ]

            retrieved_tables = fallback_tables

        # 🔹 score only relevant tables
        sorted_tables = sorted(
            retrieved_tables,
            key=lambda x: score_table(x, query),
            reverse=True
        )

        # 🔥 FINAL COMBINATION (THIS IS KEY)
        final_chunks = sorted_tables[:3] + retrieved_texts[:5]

    else:
        final_chunks = retrieved

    # =========================================================
    # 🔥 STEP 9: GENERATE ANSWER
    # =========================================================
    context_chunks = [c["text"] for c in final_chunks]

    answer = generate_answer(query, context_chunks)

    print("\nFinal Answer:\n", answer)

    # =========================================================
    # 🔍 DEBUG OUTPUT
    # =========================================================
    print("\nRetrieved Context:\n")

    for r in final_chunks:
        print("TYPE:", r["type"])
        print("PAGE:", r["page"])
        print(r["text"][:200])
        print("------")
    
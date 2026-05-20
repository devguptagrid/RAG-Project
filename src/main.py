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

    tables = extract_tables("data/ifc_report.pdf")

    print("Total tables found:", len(tables))

    # print first table
    # if tables:
        # print("\nSample table:\n")
        # for row in tables[0]["table"]:
        #     print(row)

    if tables:
        sample = tables[0]["table"]
        
        text_table = table_to_text(sample)
        
        # print("\nConverted Table:\n")
        # print(text_table)

    table_chunks = create_table_chunks(tables)

    # print("Total table chunks:", len(table_chunks))

    # print("\nSample table chunk:\n")
    # print(table_chunks[0]["text"])
    # print("\nPage:", table_chunks[0]["page"])
    # print("Type:", table_chunks[0]["type"])

    # 🔹 Step 1: text chunks (with metadata)
    text_metadata_chunks = chunk_text_with_metadata(pdf_path)

    # 🔹 Step 2: tables
    tables = extract_tables(pdf_path)
    table_chunks = create_table_chunks(tables)

    # 🔹 Step 3: merge
    all_chunks = text_metadata_chunks + table_chunks

    # print("Text chunks:", len(text_metadata_chunks))
    # print("Table chunks:", len(table_chunks))
    # print("Total chunks:", len(all_chunks))
    # print(all_chunks[0])
    # print(all_chunks[-1])   
    
    texts = [c["text"] for c in all_chunks]

    embeddings = create_embeddings(model, texts)
    index = create_faiss_index(embeddings)

    query = "What was the variance in value for 'Equity investments' under Long-Term Finance Own Account Commitments between the fiscal years 2023 and 2024?"

    indices = search_indices(query, model, index, texts, k=5)

    retrieved = [all_chunks[i] for i in indices]


    # 🔥 Detect table-type query
    if any(word in query.lower() for word in ["table", "values", "data", "numbers", "fy", "year", "income"]):

        # 🔥 IMPORTANT: use ALL tables, not retrieved ones
        all_table_chunks = [c for c in all_chunks if c["type"] == "table"]

        # score all tables
        sorted_tables = sorted(
            all_table_chunks,
            key=lambda x: score_table(x, query),
            reverse=True
        )

        # split FAISS results
        text_chunks = [c for c in retrieved if c["type"] == "text"]

        # 🔥 FINAL ORDER (THIS IS KEY)
        final_chunks = sorted_tables[:3] + text_chunks[:5]

    else:
        final_chunks = retrieved


    # 🔥 Build context
    context_chunks = [c["text"] for c in final_chunks]

    # 🔥 Generate answer
    answer = generate_answer(query, context_chunks)

    print("\nFinal Answer:\n", answer)


    # 🔍 Debug print
    print("\nRetrieved Context:\n")

    for r in final_chunks:
        print("TYPE:", r["type"])
        print("PAGE:", r["page"])
        print(r["text"][:200])
        print("------")
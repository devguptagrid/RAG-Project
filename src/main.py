from rag import *
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":

    pdf_path = "data/ifc_report.pdf"

    # 🔹 Load model
    #model = SentenceTransformer('all-MiniLM-L6-v2')

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
    # text_chunks = chunk_text_with_metadata(pdf_path)

    # # =========================================================
    # # 🔥 STEP 2: TABLE CHUNKS
    # # =========================================================
    # tables = extract_tables(pdf_path)
    # table_chunks = create_table_chunks(tables)

    # # =========================================================
    # # 🔥 STEP 3: IMAGE CHUNKS (FIXED POSITION)
    # # =========================================================
    # images = extract_images(pdf_path)

    # image_chunks = []

    # for img in images:

    #     if img["path"].endswith(".jpx"):
    #         continue

    #     try:
    #         desc = describe_image(img["path"])

    #         if is_useful_image(desc):
    #             image_chunks.append({
    #                 "text": desc,
    #                 "type": "image",
    #                 "page": img["page"]
    #             })

    #     except Exception as e:
    #         print(f"Skipping {img['path']} → {e}")

    # print("Total images:", len(images))
    # print("Useful images:", len(image_chunks))

    # # =========================================================
    # # 🔥 STEP 4: FIGURE CHUNKS (IMPORTANT ADDITION)
    # # =========================================================
    # figure_chunks = []

    # for c in text_chunks:
    #     text = c["text"].lower()

    #     if "figure" in text or "chart" in text:
    #         figure_chunks.append({
    #             "text": c["text"],
    #             "type": "image",
    #             "page": c["page"]
    #         })

    # # =========================================================
    # # 🔥 STEP 5: MERGE EVERYTHING
    # # =========================================================
    # all_chunks = text_chunks + table_chunks + image_chunks + figure_chunks

    # print("Total chunks:", len(all_chunks))

    # # =========================================================
    # # 🔥 STEP 6: EMBEDDINGS + INDEX
    # # =========================================================
    # texts = [c["text"] for c in all_chunks]

    # embeddings = create_embeddings(model, texts)
    # index = create_faiss_index(embeddings)

    # # =========================================================
    # # 🔥 STEP 7: QUERY
    # # =========================================================
    # query = "What was the total value of IFC's assets as of June 30, 2023?"

    # indices = search_indices(query, model, index, texts, k=5)

    # retrieved = [all_chunks[i] for i in indices]

    # # =========================================================
    # # 🔥 STEP 8: TABLE BOOSTING (FINAL CORRECT VERSION)
    # # =========================================================

    # if any(word in query.lower() for word in ["table", "values", "data", "numbers", "fy", "year", "income", "variance", "percentage", "%", "share", "accounted"]):

    #     # 🔹 split FAISS results
    #     retrieved_tables = [c for c in retrieved if c["type"] == "table"]
    #     retrieved_texts = [c for c in retrieved if c["type"] == "text"]

    #     # 🔥 If FAISS didn't return tables → fallback
    #     if len(retrieved_tables) == 0:

    #         # take nearby chunks (same pages as retrieved text)
    #         pages = set([c["page"] for c in retrieved_texts])

    #         fallback_tables = [
    #             c for c in table_chunks
    #             if c["page"] in pages
    #         ]

    #         retrieved_tables = fallback_tables

    #     # 🔹 score only relevant tables
    #     sorted_tables = sorted(
    #         retrieved_tables,
    #         key=lambda x: score_table(x, query),
    #         reverse=True
    #     )

    #     # 🔥 FINAL COMBINATION (THIS IS KEY)
    #     final_chunks = sorted_tables[:3] + retrieved_texts[:5]

    # else:
    #     final_chunks = retrieved

    # # =========================================================
    # # 🔥 STEP 9: GENERATE ANSWER
    # # =========================================================
    # context_chunks = [c["text"] for c in final_chunks]

    # answer = generate_answer(query, context_chunks)

    # print("\nFinal Answer:\n", answer)

    # # =========================================================
    # # 🔍 DEBUG OUTPUT
    # # =========================================================
    # print("\nRetrieved Context:\n")

    # for r in final_chunks:
    #     print("TYPE:", r["type"])
    #     print("PAGE:", r["page"])
    #     print(r["text"][:200])
    #     print("------")
    











    # # load model
    # model, processor, device = load_clip_model()

    # # patches
    # images = pdf_to_images("data/ifc_report.pdf")

    # patches = []

    # for img in images:
    #     page_patches = create_image_patches(
    #         image_path=img["path"],
    #         page_number=img["page"]
    #     )
    #     patches.extend(page_patches)

    # # test small subset (important)
    # patches_small = patches

    # # embeddings
    # patch_embeddings = embed_image_patches(model, processor, patches_small, device)

    # # query
    # query = "What sector has 2,332 million $ disbursed investment?"


    # # retrieval
    # #retrieved = retrieve_top_patches(query_embedding, patch_embeddings, patches_small, top_k=5)
    # retrieved = retrieve_patches_maxsim(
    #     model,
    #     processor,
    #     query,
    #     patches_small,
    #     patch_embeddings,
    #     device,
    #     top_k=5
    # )

    # print("Retrieved patches:", len(retrieved))

    # # show them
    # for i, p in enumerate(retrieved):
    #     print(f"\nPatch {i+1} | Page {p['page']} | Coords {p['coords']}")
    #     p["image"].show()




    # =========================================================
    # 🔥 STEP 1: TEXT CHUNKS (COARSE RETRIEVAL)
    # =========================================================
    # pdf_path = "data/ifc_report.pdf"

    # print("Creating text chunks...")
    # text_chunks = chunk_text_with_metadata(pdf_path)

    # texts = [c["text"] for c in text_chunks]

    # print("Creating embeddings...")
    # text_model = load_hybrid_models()[4]   # SentenceTransformer only

    # text_embeddings = create_embeddings(text_model, texts)
    # index = create_faiss_index(text_embeddings)

    # # =========================================================
    # # 🔥 STEP 2: QUERY
    # # =========================================================
    # query = "With which accounting principles do IFC's consolidated financial statements conform?"

    # print("\nQuery:", query)

    # # =========================================================
    # # 🔥 STEP 3: GET TOP PAGES (CRITICAL FIX)
    # # =========================================================
    # indices = search_indices(query, text_model, index, texts, k=5)
    # retrieved_text = [text_chunks[i] for i in indices]

    # top_pages = get_top_pages_hybrid(
    #     query,
    #     text_chunks,
    #     text_model,
    #     index,
    #     texts,
    #     k=5
    # )

    # print("\nTop pages from text retrieval:", top_pages)

    # # =========================================================
    # # 🔥 STEP 4: LOAD IMAGES
    # # =========================================================
    # images = pdf_to_images(pdf_path)

    # # 🔥 FILTER ONLY RELEVANT PAGES
    # filtered_images = [img for img in images if img["page"] in top_pages]

    # print("Filtered pages:", [img["page"] for img in filtered_images])

    # # =========================================================
    # # 🔥 STEP 5: CREATE PATCHES (ONLY RELEVANT PAGES)
    # # =========================================================
    # patches = []

    # for img in filtered_images:
    #     page_patches = create_image_patches(
    #         img["path"],
    #         img["page"]
    #     )
    #     patches.extend(page_patches)

    # print("Total filtered patches:", len(patches))

    # # 🔥 LIMIT (VERY IMPORTANT)
    # patches_small = patches

    # print("Using patches:", len(patches_small))

    # # =========================================================
    # # 🔥 STEP 6: LOAD HYBRID MODELS
    # # =========================================================
    # clip_model, clip_processor, blip_model, blip_processor, text_model, device = load_hybrid_models()

    # # =========================================================
    # # 🔥 STEP 7: EMBED PATCHES (HYBRID)
    # # =========================================================
    # print("\nEmbedding patches (hybrid)...")

    # patches_small = attach_text_to_patches(patches_small, text_chunks)

    # patch_embeddings, captions = embed_all_patches_hybrid(
    #     clip_model,
    #     clip_processor,
    #     blip_model,
    #     blip_processor,
    #     text_model,
    #     patches_small,
    #     device
    # )

    # # =========================================================
    # # 🔥 STEP 8: QUERY EMBEDDING
    # # =========================================================
    # query_emb = embed_query_hybrid(
    #     clip_model,
    #     clip_processor,
    #     text_model,
    #     query,
    #     device
    # )

    # # =========================================================
    # # 🔥 STEP 9: RETRIEVAL
    # # =========================================================
    # retrieved_patches = retrieve_hybrid(
    #     query_emb,
    #     patch_embeddings,
    #     patches_small,
    #     query,
    #     captions,
    #     top_k=5
    # )

    # # =========================================================
    # # 🔥 STEP 10: OUTPUT
    # # =========================================================
    # print("\nRetrieved patches:\n")

    # for i, p in enumerate(retrieved_patches):
    #     print(f"Patch {i+1} | Page {p['page']} | Coords {p['coords']}")
    #     p["image"].show()

    # answer = get_final_answer(query, text_chunks, top_pages)

    # print("\nFinal Answer:", answer)


    ##################################################

    # # =========================================================
    # # 🔥 STEP 0: QUERY (DEFINE FIRST)
    # # =========================================================
    # query = "What was IFC's Net Income for the fiscal year ending June 30, 2024?"

    # # =========================================================
    # # 🔥 STEP 1: LOAD MODEL (PaliGemma)
    # # =========================================================
    # model, processor, device = load_paligemma_model()

    # # =========================================================
    # # 🔥 STEP 2: TEXT PIPELINE (REQUIRED FOR HYBRID RETRIEVAL)
    # # =========================================================
    # pdf_path = "data/ifc_report.pdf"

    # print("\nCreating text chunks...")
    # text_chunks = chunk_text_with_metadata(pdf_path)
    # texts = [c["text"] for c in text_chunks]

    # print("Creating text embeddings...")
    # text_model = SentenceTransformer("all-MiniLM-L6-v2")

    # text_embeddings = create_embeddings(text_model, texts)
    # index = create_faiss_index(text_embeddings)

    # # =========================================================
    # # 🔥 STEP 3: HYBRID PAGE RETRIEVAL
    # # =========================================================
    # print("\nRunning hybrid retrieval...")

    # top_pages = get_top_pages_hybrid(
    #     query,
    #     text_chunks,
    #     text_model,
    #     index,
    #     texts,
    #     k=10   # 🔥 increased
    # )

    # print("Top pages:", top_pages)

    # # 🔥 DEBUG
    # print("\nDEBUG TOP PAGES:")
    # for p in top_pages:
    #     print("Page:", p)

    # # 🔥 SAFETY INJECTION
    # if 8 not in top_pages:
    #     print("⚠️ Injecting page 8 manually")
    #     top_pages.append(8)

    

    # # DEBUG check
    # if 8 in top_pages:
    #     print("✅ Page 8 FOUND")
    # else:
    #     print("❌ Page 8 MISSING")

    # # =========================================================
    # # 🔥 STEP 4: LOAD IMAGES + FILTER
    # # =========================================================
    # images = pdf_to_images(pdf_path)

    # filtered_images = [img for img in images if img["page"] in top_pages]

    # print("Filtered pages:", [img["page"] for img in filtered_images])

    # # =========================================================
    # # 🔥 STEP 5: CREATE PATCHES (ONLY RELEVANT PAGES)
    # # =========================================================
    # patches = []

    # for img in filtered_images:
    #     patches.extend(create_image_patches(img["path"], img["page"]))

    # print("Total patches:", len(patches))

    # # 🔥 LIMIT HARD (VERY IMPORTANT)
    # patches_small = patches[:400]

    # # =========================================================
    # # 🔥 STEP 6: EMBED PATCHES (TOKEN LEVEL)
    # # =========================================================
    # print("\nEmbedding patch tokens...")

    # patch_token_embeddings = []

    # for i, p in enumerate(patches_small):
    #     print(f"Embedding patch {i}")

    #     tokens = embed_patch_tokens_paligemma(
    #         model,
    #         processor,
    #         p,
    #         device
    #     )

    #     patch_token_embeddings.append(tokens)

    # # =========================================================
    # # 🔥 STEP 7: QUERY TOKEN EMBEDDING
    # # =========================================================
    # tokenizer, text_token_model = load_text_token_model()

    # query_tokens = embed_query_tokens_text(
    #     query,
    #     tokenizer,
    #     text_token_model
    # )

    # # 🔥 PROJECT TO MATCH DIMENSION
    # query_tokens = project_query_to_patch_dim(query_tokens)

    # # =========================================================
    # # 🔥 STEP 8: MAXSIM RETRIEVAL
    # # =========================================================
    # retrieved_patches = retrieve_patches_maxsim_paligemma(
    #     query_tokens,
    #     patch_token_embeddings,
    #     patches_small,
    #     top_k=3
    # )

    # print("\nRetrieved patches:\n")

    # for i, p in enumerate(retrieved_patches):
    #     print(f"Patch {i+1} | Page {p['page']} | Coords {p['coords']}")
    #     p["image"].show()
    # from collections import Counter

    # # =========================================================
    # # 🔥 STEP 8.5: FINAL PAGE SELECTION (CRITICAL FIX)
    # # =========================================================

    # # pages from patches
    # patch_pages = [p["page"] for p in retrieved_patches]

    # patch_page_counts = Counter(patch_pages)

    # # combine candidates
    # candidate_pages = top_pages + list(patch_page_counts.keys())

    # final_scores = {}

    # for page in candidate_pages:
    #     score = 0

    #     # 🔥 VERY STRONG hybrid boost (MOST IMPORTANT)
    #     if page in top_pages:
    #         score += (len(top_pages) - top_pages.index(page)) * 5

    #     # 🔥 weaker patch influence
    #     score += patch_page_counts.get(page, 0) * 1

    #     # 🔥 keyword boost (strong)
    #     page_text = " ".join([c["text"] for c in text_chunks if c["page"] == page])

    #     for word in query.lower().split():
    #         if word in page_text.lower():
    #             score += 3

    #     final_scores[page] = score

    # # select best page
    # best_page = max(final_scores, key=final_scores.get)

    # print("\nFinal selected page:", best_page)
    # # =========================================================
    # # 🔥 STEP 9: FINAL ANSWER (FULL PAGE, NOT PATCH)
    # # =========================================================
    # answer = generate_answer_paligemma(
    #     model,
    #     processor,
    #     query,
    #     best_page,
    #     filtered_images,   # 🔥 IMPORTANT
    #     top_pages,
    #     device
    # )

    # print("\nFinal Answer:\n", answer)












    ##########################gemini



from rag import *
from sentence_transformers import SentenceTransformer
import os

if __name__ == "__main__":
    
    # =============================================================================
    # =============================================================================
    #
    #   PHASE 6 — END-TO-END MULTIMODAL RAG (ColPali-like Approach)
    #
    #   PaliGemma 3B for embeddings  +  Gemini 2.5 Pro for generation
    #   Overlapping patches • MaxSim late interaction • Hybrid reranking
    #   Multi-page visual context • Enhanced source attribution
    #
    # =============================================================================
    # =============================================================================

    print("\n\n" + "="*60)
    print("🚀 PHASE 6: END-TO-END MULTIMODAL RAG")
    print("="*60)

    # =========================================================
    # 🔥 STEP 0: QUERY
    # =========================================================
    query_p6 = "What was the value of predominant currency in IFC's disbursed loan portfolio as of June 30, 2024?"
    pdf_path_p6 = "data/ifc_report.pdf"

    print(f"\nQuery: '{query_p6}'")

    # =========================================================
    # 🔥 STEP 1: RUN FULL PHASE 6 PIPELINE
    # =========================================================
    result_p6 = run_phase6_pipeline(
        query=query_p6,
        pdf_path=pdf_path_p6,
        top_k_pages=3,
        max_patches=400
    )

    # =========================================================
    # 🔥 STEP 2: DISPLAY RESULTS
    # =========================================================
    print("\n" + "="*60)
    print("🔥 PHASE 6 — FINAL ANSWER:")
    print("="*60)
    print(result_p6["answer"])

    print(f"\n📄 Retrieved pages: {result_p6['final_pages']}")
    print(f"🧩 Patches processed: {result_p6['num_patches_processed']}")

    print("\n⏱  Timing breakdown:")
    for step, duration in result_p6["timings"].items():
        print(f"    {step}: {duration:.1f}s")

    print(f"\n📊 Attribution images:")
    for attr in result_p6["attribution_paths"]:
        print(f"    Page {attr['page']}: {attr['path']} ({attr['num_patches']} patches)")
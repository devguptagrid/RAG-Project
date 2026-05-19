from rag import *
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    pdf_path = "data/ifc_report.pdf"
    
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    chunks = filter_chunks(chunks)
    
    print("Total chunks:", len(chunks))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = create_embeddings(model,chunks)
    
    print("\nEmbedding shape:", embeddings.shape)

    index = create_faiss_index(embeddings)
    
    # test query
    query = "What is the mission of IFC?"
    
    retrieved = search(query, model, index, chunks, k=10)

    results = rerank(query, retrieved)
    

    answer = generate_answer(query, results)

    print("\nFinal Answer:\n", answer)



    client = create_qdrant_client()

    collection_name = "ifc_collection"

    create_collection(client, collection_name, embeddings.shape[1])

    insert_into_qdrant(client, collection_name, chunks, embeddings)

    # test query
    query = "What is the mission of IFC?"

    retrieved = search_qdrant(client, collection_name, query, model, k=10)
    results = rerank(query, retrieved)
    print("\nQdrant Results:\n")

    for i, res in enumerate(results):
        print(f"\nResult {i+1}:\n", res)
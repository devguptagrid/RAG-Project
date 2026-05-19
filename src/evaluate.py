import pandas as pd
from rag import *
from sentence_transformers import SentenceTransformer


# 🔹 Load dataset
df = pd.read_excel("data/RAG_evaluation_dataset.xlsx")


# 🔹 Load your RAG pipeline (same as before)
pdf_path = "data/ifc_report.pdf"

text = extract_text_from_pdf(pdf_path)
chunks = filter_chunks(chunk_text(text))

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = create_embeddings(model, chunks)

index = create_faiss_index(embeddings)


# 🔹 Store results
results_data_faiss = []
results_data_qdrant = []


client = create_qdrant_client()
collection_name = "ifc_collection"

create_collection(client, collection_name, embeddings.shape[1])
insert_into_qdrant(client, collection_name, chunks, embeddings)

# 🔹 Loop through dataset
for i, row in df.iterrows():
    
    question = row["Question"]
    ground_truth = row["Ground_Truth_Answer"]
    
    # 🔹 Step 1: Retrieve
    retrieved_faiss = search_text(question, model, index, chunks, k=10)
    
    retrieved_qdrant = search_qdrant(client, collection_name, question, model, k=10)
    

    # 🔹 Step 2: Generate answer
    answer_faiss = generate_answer(question, retrieved_faiss)
    answer_qdrant = generate_answer(question, retrieved_qdrant)
    

    
    # 🔹 Save result
    results_data_faiss.append({
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer_faiss,
        "contexts": retrieved_faiss
    })
    results_data_qdrant.append({
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer_qdrant,
        "contexts": retrieved_qdrant
    })


# 🔹 Convert to DataFrame
results_df_faiss = pd.DataFrame(results_data_faiss)
results_df_qdrant = pd.DataFrame(results_data_qdrant)


# 🔹 Save results
results_df_faiss.to_csv("results_faiss.csv", index=False)
results_df_qdrant.to_csv("results_qdrant.csv", index=False)

print("Evaluation data generated and saved to results_faiss.csv and results_qdrant.csv")
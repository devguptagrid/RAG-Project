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
results_data = []


# 🔹 Loop through dataset
for i, row in df.iterrows():
    
    question = row["Question"]
    ground_truth = row["Ground_Truth_Answer"]
    
    # 🔹 Step 1: Retrieve
    retrieved = search(question, model, index, chunks, k=3)
    reranked = rerank(
        question,
        retrieved,
        model,
        embeddings,
        chunks
    )[:3]   # keep top 5
    
    # 🔹 Step 2: Generate answer
    answer = generate_answer(question, reranked)
    
    # 🔹 Save result
    results_data.append({
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "contexts": reranked
    })


# 🔹 Convert to DataFrame
results_df = pd.DataFrame(results_data)


# 🔹 Save results
results_df.to_csv("results.csv", index=False)

print("Evaluation data generated and saved to results.csv")
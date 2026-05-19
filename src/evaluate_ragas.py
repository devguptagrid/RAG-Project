import pandas as pd
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_google_vertexai import VertexAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

llm = VertexAI(
    model_name="gemini-2.0-flash",   # safer version
    project="gd-gcp-gridu-genai",
    location="us-central1"
)

ragas_llm = LangchainLLMWrapper(llm)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)


# 🔹 Load results
df = pd.read_csv("results.csv")


ragas_llm = LangchainLLMWrapper(llm)

# 🔹 Convert contexts from string → list
import ast
df["contexts"] = df["contexts"].apply(ast.literal_eval)


# 🔹 Convert to dataset format
dataset = Dataset.from_pandas(df.rename(columns={
    "question": "question",
    "answer": "answer",
    "ground_truth": "ground_truth",
    "contexts": "contexts"
}))


# 🔹 Run evaluation
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)


print(result)
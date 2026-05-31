"""
Phase 6 — Pipeline Comparison Script
=====================================
Runs Phase 5 (existing) and Phase 6 (new ColPali-like) pipelines on the
evaluation dataset and compares results side-by-side.

Usage:
    python phase6_pipeline.py
"""

import pandas as pd
from rag import (
    run_phase5_pipeline,
    run_phase6_pipeline,
    compare_pipelines
)


def main():
    pdf_path = "data/ifc_report.pdf"

    # =========================================================
    # Load evaluation dataset
    # =========================================================
    print("Loading evaluation dataset...")
    df = pd.read_excel("data/RAG_evaluation_dataset.xlsx")

    queries = df["Question"].tolist()
    ground_truths = df["Ground_Truth_Answer"].tolist()

    # Limit to first 5 queries for initial testing (full dataset takes long)
    test_queries = queries[:5]

    print(f"Running comparison on {len(test_queries)} queries...")
    print("=" * 60)

    # =========================================================
    # Run comparison
    # =========================================================
    comparison_df = compare_pipelines(test_queries, pdf_path)

    # Add ground truth
    comparison_df["ground_truth"] = ground_truths[:len(test_queries)]

    # =========================================================
    # Save results
    # =========================================================
    output_file = "phase6_comparison.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\n✅ Comparison saved to {output_file}")

    # =========================================================
    # Print summary
    # =========================================================
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for i, row in comparison_df.iterrows():
        print(f"\n📌 Query {i+1}: {row['query'][:80]}...")
        print(f"   Ground Truth: {row['ground_truth'][:100]}...")
        print(f"   Phase 5 → Pages {row['phase5_pages']} | Time: {row['phase5_time_s']}s")
        print(f"   Phase 5 Answer: {row['phase5_answer'][:100]}...")
        print(f"   Phase 6 → Pages {row['phase6_pages']} | Time: {row['phase6_time_s']}s")
        print(f"   Phase 6 Answer: {row['phase6_answer'][:100]}...")
        print("-" * 60)

    # Timing comparison
    avg_p5 = comparison_df["phase5_time_s"].mean()
    avg_p6 = comparison_df["phase6_time_s"].mean()

    print(f"\n⏱  Average Phase 5 time: {avg_p5:.1f}s")
    print(f"⏱  Average Phase 6 time: {avg_p6:.1f}s")

    if avg_p6 > avg_p5:
        print(f"   Phase 6 is {avg_p6/avg_p5:.1f}x slower (expected: more patches + multi-page)")
    else:
        print(f"   Phase 6 is {avg_p5/avg_p6:.1f}x faster")


if __name__ == "__main__":
    main()

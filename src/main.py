"""
Multimodal Financial Report RAG - Entry Point

Usage:
    python -m src.main                          # Run with default example queries
    python -m src.main "Your question here"     # Run with a custom question
"""

import sys

from src.parser import parse_pdf
from src.indexer import build_index
from src.retriever import build_rag_graph


def run_pipeline(questions: list[str] | None = None):
    """Parse, index, and query the Samsung financial report."""

    if questions is None:
        questions = [
            "What was Samsung's total revenue in Q4?",
            "How did the semiconductor division perform compared to previous quarters?",
            "What are the key financial highlights shown in the charts?",
        ]

    # Step 1: Parse PDF
    print("=" * 60)
    print("STEP 1: Parsing PDF with LlamaParse...")
    print("=" * 60)
    text_chunks, image_records = parse_pdf()

    # Step 2: Build multi-vector index
    print("\n" + "=" * 60)
    print("STEP 2: Building multi-vector index...")
    print("=" * 60)
    retriever = build_index(text_chunks, image_records)

    # Step 3: Build LangGraph RAG pipeline
    print("\n" + "=" * 60)
    print("STEP 3: Building LangGraph RAG pipeline...")
    print("=" * 60)
    rag_app = build_rag_graph(retriever)
    print("Pipeline ready.")

    # Step 4: Run queries
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"QUERY {i}: {question}")
        print("=" * 60)

        result = rag_app.invoke({
            "question": question,
            "context_texts": [],
            "context_image_paths": [],
            "answer": "",
        })

        print(f"\nANSWER:\n{result['answer']}")
        print(
            f"\n[Retrieved {len(result['context_texts'])} text chunks, "
            f"{len(result['context_image_paths'])} images]"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
        run_pipeline([user_question])
    else:
        run_pipeline()

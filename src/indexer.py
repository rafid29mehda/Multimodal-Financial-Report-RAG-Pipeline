import uuid
import time
import base64
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.config import (
    EMBEDDING_MODEL,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    ID_KEY,
    GROQ_API_KEY,
    LLM_MODEL,
    TOP_K,
)


def _get_embedding_function():
    """Initialize the local HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def _summarize_image(image_path: str, max_retries: int = 5) -> str:
    """Use Groq (Llama 3.2 Vision) to generate a text summary of a chart/table image.

    Includes retry logic with exponential backoff for rate limits.
    """
    llm = ChatGroq(
        model_name=LLM_MODEL,
        api_key=GROQ_API_KEY,
    )

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(ext, "image/png")

    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "Describe this financial chart or table in detail. "
                "Include all data points, trends, labels, and numbers visible. "
                "This description will be used for semantic search."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
        },
    ])

    for attempt in range(max_retries):
        try:
            response = llm.invoke([message])
            return response.content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait_time = 15 * (2 ** attempt)  # 15s, 30s, 60s, 120s, 240s
                print(f"    Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Failed to summarize {image_path} after {max_retries} retries")


def build_index(
    text_chunks: list[dict], image_records: list[dict]
) -> MultiVectorRetriever:
    """Build the multi-vector index from parsed text chunks and images.

    Text chunks are embedded directly (text IS the summary).
    Images are summarized by Llama 3.2 Vision (via Groq), summaries are embedded, and raw
    image paths are stored in the docstore for later retrieval.

    Args:
        text_chunks: from parser, list of {"page": int, "content": str}
        image_records: from parser, list of {"path": str, "name": str, "page": int}

    Returns:
        A configured MultiVectorRetriever
    """
    print("Loading embedding model...")
    embeddings = _get_embedding_function()

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    docstore = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
        search_kwargs={"k": TOP_K},
    )

    # ---- Index text chunks ----
    text_doc_ids = [str(uuid.uuid4()) for _ in text_chunks]
    summary_docs = []
    original_docs = []

    for i, chunk in enumerate(text_chunks):
        summary_docs.append(
            Document(
                page_content=chunk["content"],
                metadata={
                    ID_KEY: text_doc_ids[i],
                    "type": "text",
                    "page": chunk["page"],
                },
            )
        )
        original_docs.append(
            Document(
                page_content=chunk["content"],
                metadata={"type": "text", "page": chunk["page"]},
            )
        )

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(text_doc_ids, original_docs)))
    print(f"Indexed {len(text_chunks)} text chunks.")

    # ---- Index images ----
    if image_records:
        img_doc_ids = [str(uuid.uuid4()) for _ in image_records]

        for i, img in enumerate(image_records):
            print(f"Summarizing image {i + 1}/{len(image_records)}: {img['name']}")
            summary_text = _summarize_image(img["path"])

            # Embed the summary in the vectorstore
            retriever.vectorstore.add_documents([
                Document(
                    page_content=summary_text,
                    metadata={
                        ID_KEY: img_doc_ids[i],
                        "type": "image",
                        "page": img["page"],
                        "image_name": img["name"],
                    },
                )
            ])

            # Store the raw image path in the docstore
            retriever.docstore.mset([
                (
                    img_doc_ids[i],
                    Document(
                        page_content=img["path"],
                        metadata={
                            "type": "image",
                            "page": img["page"],
                            "image_path": img["path"],
                        },
                    ),
                )
            ])

        print(f"Indexed {len(image_records)} images.")

    return retriever

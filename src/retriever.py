import base64
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langgraph.graph import StateGraph, START, END

from src.config import GROQ_API_KEY, LLM_MODEL


# ---- State Definition ----

class RAGState(TypedDict):
    question: str
    context_texts: list[str]
    context_image_paths: list[str]
    answer: str


# ---- Node Functions ----

def retrieve_node(state: RAGState, *, retriever: MultiVectorRetriever) -> dict:
    """Retrieve relevant text chunks and image paths from the index."""
    question = state["question"]
    docs = retriever.invoke(question)

    context_texts = []
    context_image_paths = []

    for doc in docs:
        if doc.metadata.get("type") == "image":
            image_path = doc.metadata.get("image_path", doc.page_content)
            if Path(image_path).exists():
                context_image_paths.append(image_path)
        else:
            context_texts.append(doc.page_content)

    return {
        "context_texts": context_texts,
        "context_image_paths": context_image_paths,
    }


def generate_node(state: RAGState) -> dict:
    """Generate an answer using Llama 3.2 Vision (via Groq) with text context and images."""
    llm = ChatGroq(
        model_name=LLM_MODEL,
        api_key=GROQ_API_KEY,
    )

    content_parts = []

    # System instruction
    content_parts.append({
        "type": "text",
        "text": (
            "You are a financial analyst assistant. Answer the question based on "
            "the provided context from the Samsung Electronics 4Q financial report. "
            "If charts or images are provided, analyze them carefully. "
            "Cite specific numbers and data points when available.\n\n"
        ),
    })

    # Text context
    if state["context_texts"]:
        combined_text = "\n\n---\n\n".join(state["context_texts"])
        content_parts.append({
            "type": "text",
            "text": f"TEXT CONTEXT:\n{combined_text}\n\n",
        })

    # Image context (Groq limits to 5 images per request)
    for img_path in state["context_image_paths"][:5]:
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(img_path).suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        mime_type = mime_map.get(ext, "image/png")
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{img_b64}"},
        })

    # The actual question
    content_parts.append({
        "type": "text",
        "text": f"QUESTION: {state['question']}",
    })

    message = HumanMessage(content=content_parts)
    response = llm.invoke([message])

    return {"answer": response.content}


def build_rag_graph(retriever: MultiVectorRetriever):
    """Build and compile the LangGraph RAG workflow.

    Graph: START -> retrieve -> generate -> END
    """
    graph = StateGraph(RAGState)

    # Bind retriever to the retrieve node via closure
    def _retrieve(state: RAGState) -> dict:
        return retrieve_node(state, retriever=retriever)

    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

# Multimodal Financial Report RAG Pipeline

A **multi-vector retrieval-augmented generation** system that analyzes financial report PDFs containing charts, tables, and graphs. Built with LangGraph, LlamaParse, ChromaDB, and Llama 4 Scout (via Groq).

## Problem

Standard RAG pipelines extract only text from PDFs, causing hallucinated answers when critical data lives in charts or tables. This system uses **multi-vector retrieval** — embedding text summaries of images for search, but retrieving the raw images for the LLM to interpret visually.

## Architecture

```
PDF -> [LlamaParse] -> text chunks + chart/table images
                            |                |
                      embed text      Llama 4 Scout summarizes -> embed summaries
                            |                |
                      ChromaDB (vectorstore: all embeddings, keyed by doc_id)
                      InMemoryStore (docstore: raw text / image paths by doc_id)

Query -> [retrieve node] -> similarity search -> fetch raw content from docstore
       -> [generate node] -> multimodal prompt (text + images) -> Llama 4 Scout -> answer
```

| Component | Tool |
|-----------|------|
| PDF Parsing | LlamaParse | 
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | 
| Vector Store | ChromaDB | 
| Orchestration | LangGraph | 
| Generation | Llama 4 Scout via Groq | 

## Setup

### 1. API Keys

Add API Keys to `.env`:
```
LLAMA_CLOUD_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### 2. Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add a PDF

Place a financial report PDF in the project root. The default expected file is `SAMSUNG ELECTRONICS_4Q_financial_report.pdf` (configurable in `src/config.py`).

## Usage

```bash
# Run with default example queries
python -m src.main

# Run with a custom question
python -m src.main "What was Samsung's operating profit in Q4?"
```

The pipeline runs in 4 steps:
1. **Parse** — LlamaParse extracts text and images from the PDF
2. **Index** — Text chunks are embedded directly; images are summarized by Llama 4 Scout (via Groq) then embedded. Raw content is stored in a docstore linked by ID.
3. **Build graph** — LangGraph compiles the retrieve -> generate workflow
4. **Query** — Retrieves relevant text/images and sends them to Llama 4 Scout for a multimodal answer

## Project Structure

```
src/
  config.py      - Paths, API keys, model names, retrieval parameters
  parser.py      - PDF parsing with LlamaParse (text + image extraction)
  indexer.py     - Multi-vector indexing (ChromaDB + InMemoryStore)
  retriever.py   - LangGraph RAG workflow (retrieve -> generate nodes)
  main.py        - Entry point
```

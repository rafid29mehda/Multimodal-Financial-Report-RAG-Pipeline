import os

from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    LLAMA_CLOUD_API_KEY,
    PDF_PATH,
    IMAGE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def parse_pdf() -> tuple[list[dict], list[dict]]:
    """Parse the Samsung PDF into text chunks and extracted images.

    Returns:
        text_chunks: list of {"page": int, "content": str}
        image_records: list of {"path": str, "name": str, "page": int}
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=True,
    )

    # Get structured JSON with page-level markdown text
    print("Sending PDF to LlamaParse API...")
    json_results = parser.get_json_result(str(PDF_PATH))
    pages = json_results[0]["pages"]
    print(f"Received {len(pages)} pages of parsed text.")

    # Extract images (charts, tables, figures) to disk
    print("Extracting images from PDF...")
    image_dicts = parser.get_images(json_results, download_path=str(IMAGE_DIR))
    print(f"Extracted {len(image_dicts)} images.")

    # Chunk the text for indexing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    text_chunks = []
    for page in pages:
        page_num = page.get("page", 0)
        page_md = page.get("md", "")
        if not page_md.strip():
            continue
        chunks = splitter.split_text(page_md)
        for chunk in chunks:
            text_chunks.append({"page": page_num, "content": chunk})

    # Normalize image records
    image_records = []
    for img in image_dicts:
        image_records.append({
            "path": img["path"],
            "name": img.get("name", os.path.basename(img["path"])),
            "page": img.get("page", -1),
        })

    # Fallback: if no images were extracted, use page screenshots
    if not image_records:
        print("No images extracted. Retrying with page screenshots...")
        screenshot_parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            take_screenshot=True,
            verbose=True,
        )
        screenshot_results = screenshot_parser.get_json_result(str(PDF_PATH))
        screenshot_images = screenshot_parser.get_images(
            screenshot_results, download_path=str(IMAGE_DIR)
        )
        for img in screenshot_images:
            image_records.append({
                "path": img["path"],
                "name": img.get("name", os.path.basename(img["path"])),
                "page": img.get("page", -1),
            })
        print(f"Captured {len(image_records)} page screenshots as fallback.")

    print(f"Total: {len(text_chunks)} text chunks, {len(image_records)} images.")
    return text_chunks, image_records

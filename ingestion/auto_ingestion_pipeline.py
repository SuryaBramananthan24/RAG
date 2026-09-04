import io
import base64
from pathlib import Path

from PIL import Image as PILImage
from transformers import pipeline

from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "ChromaDB"
PDF_COLLECTION = "Pdf"
URL_COLLECTION = "Url"
MAX_CHARACTERS = 600
OVERLAP = 75


embedding_model = HuggingFaceEmbeddings()
image_captioner = pipeline(
    "image-to-text",
    model="nlpconnect/vit-gpt2-image-captioning",
    device=-1
)

def get_vector_store(collection_name):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )


def get_collection_name(file_path):
    return PDF_COLLECTION


def partition_file(file_path):
    print(f"\nReading file: {file_path}")

    return partition(
        filename=file_path,
        strategy="fast",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        infer_table_structure=True,
        include_metadata=True
    )


def partition_url(url):
    print(f"\nReading URL: {url}")

    return partition(
        url=url,
        include_metadata=True
    )


def get_image_caption(element):
    try:
        metadata = element.metadata.to_dict()
        image_base64 = metadata.get("image_base64")

        if not image_base64:
            return None

        image_bytes = base64.b64decode(image_base64)

        image = PILImage.open(
            io.BytesIO(image_bytes)
        )
        result = image_captioner(image)

        if result:
            return result[0]["generated_text"]

    except Exception as error: 
        print(f"Unable to process image: {error}")
    return None


def process_elements(elements):
    processed_elements = []

    for element in elements:
        if getattr(element, "category", None) == "Image":
            caption = get_image_caption(element)

            if caption:
                print(f"Image caption: {caption}")

                try:
                    element.text = caption
                except Exception:
                    pass

        processed_elements.append(element)

    return processed_elements


def create_documents(elements, source, source_type):
    chunks = chunk_elements(
        elements,
        max_characters=MAX_CHARACTERS,
        overlap=OVERLAP
    )

    documents = []

    for chunk in chunks:
        text = getattr(chunk, "text", "")

        if not text or not text.strip():
            continue

        metadata = {
            "source": source,
            "source_type": source_type
        }

        chunk_metadata = chunk.metadata

        page_number = getattr(
            chunk_metadata,
            "page_number",
            None
        )

        filetype = getattr(
            chunk_metadata,
            "filetype",
            None
        )

        if page_number:
            metadata["page_number"] = page_number

        if filetype:
            metadata["filetype"] = filetype

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    return documents


def store_documents(documents, collection_name):
    if not documents:
        print("No documents available for storage.")
        return

    vector_store = get_vector_store(collection_name)

    vector_store.add_documents(documents)

    print(
        f"Stored {len(documents)} chunks "
        f"in '{collection_name}' collection."
    )


def ingest_file(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"File not found: {file_path}")
        return

    elements = partition_file(str(path))
    elements = process_elements(elements)

    documents = create_documents(
        elements=elements,
        source=path.name,
        source_type="file"
    )

    collection_name = get_collection_name(file_path)

    store_documents(
        documents,
        collection_name
    )


def ingest_url(url):
    elements = partition_url(url)
    elements = process_elements(elements)

    documents = create_documents(
        elements=elements,
        source=url,
        source_type="url"
    )

    store_documents(
        documents,
        URL_COLLECTION
    )


def ingest_sources(file_paths=None, urls=None):
    file_paths = file_paths or []
    urls = urls or []

    print("Starting ingestion...")

    for file_path in file_paths:
        ingest_file(file_path)

    for url in urls:
        ingest_url(url)

    print("Ingestion completed.")
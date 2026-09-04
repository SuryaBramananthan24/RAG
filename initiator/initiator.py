from pathlib import Path

from auto_ingestion_pipeline import ingest_sources
from auto_retrieval_pipeline import retrieve_and_answer
from auto_rag_evaluator import run_evaluation

BASE_DIR = Path(**file**).resolve().parent.parent

FILES_TO_INGEST = [BASE_DIR / "data" / "documents" / "rag.pdf"]
URLS_TO_INGEST = ["https://www.preparationtech.com/post/black-tech-history-month-2021"]
RUN_RETRIEVAL_TEST = True
RUN_EVALUATION = True

def main():
print("\n" + "=" * 60)
print("AUTOMATED RAG PIPELINE")
print("=" * 60)


valid_files = []

for file_path in FILES_TO_INGEST:
    if file_path.exists():
        print(f"Found file: {file_path.name}")
        valid_files.append(str(file_path))
    else:
        print(f"File not found: {file_path}")

print("\n" + "=" * 60)
print("INGESTING SOURCES")
print("=" * 60)

ingest_sources(
    file_paths=valid_files,
    urls=URLS_TO_INGEST
)

if RUN_RETRIEVAL_TEST:
    print("\n" + "=" * 60)
    print("TESTING RETRIEVAL")
    print("=" * 60)

    query = (
        "Retrieve the paragraph where recurrent models "
        "perform computations sequentially."
    )

    answer, documents = retrieve_and_answer(
        query=query,
        collection_name="Pdf"
    )

    print(f"\nQuery: {query}")

    print("\n" + "-" * 60)
    print("GENERATED ANSWER")
    print("-" * 60)

    print(answer)

    print("\n" + "-" * 60)
    print("RETRIEVED SOURCES")
    print("-" * 60)

    for index, document in enumerate(documents, start=1):
        print(f"\nResult {index}")
        print(document.page_content)

        if document.metadata:
            print(f"\nMetadata: {document.metadata}")

if RUN_EVALUATION:
    print("\n" + "=" * 60)
    print("RUNNING RAG EVALUATION")
    print("=" * 60)

    run_evaluation()

print("\n" + "=" * 60)
print("PIPELINE COMPLETED")
print("=" * 60)

if **name** == "**main**":
main()

import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings

from auto_retrieval_pipeline import (
    get_vector_store,
    PDF_COLLECTION,
    URL_COLLECTION
)


embedding_model = HuggingFaceEmbeddings()


def calculate_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / denominator
    )


def evaluate_collection(
    collection_name,
    tests,
    k=3
):
    vector_store = get_vector_store(
        collection_name
    )

    results = []

    for test in tests:
        query = test["query"]

        expected_answer = (
            test["expected_answer_snippet"]
            .lower()
        )

        print("\n" + "-" * 60)
        print(f"Evaluating: {query}")

        documents = vector_store.similarity_search(
            query,
            k=k
        )

        if not documents:
            print("No documents retrieved.")

            results.append({
                "query": query,
                "hit": False,
                "similarity": 0.0
            })

            continue

        chunks = [
            document.page_content
            for document in documents
        ]

        hit = any(
            expected_answer in chunk.lower()
            for chunk in chunks
        )

        query_embedding = (
            embedding_model.embed_query(query)
        )

        chunk_embedding = (
            embedding_model.embed_query(
                chunks[0]
            )
        )

        similarity = calculate_similarity(
            query_embedding,
            chunk_embedding
        )

        print(f"Hit: {hit}")
        print(
            f"Similarity: {similarity:.4f}"
        )

        results.append({
            "query": query,
            "hit": hit,
            "similarity": similarity
        })

    return results


def print_summary(results):
    if not results:
        print("No evaluation results.")
        return

    total = len(results)

    successful = sum(
        result["hit"]
        for result in results
    )

    accuracy = (
        successful / total
    ) * 100

    average_similarity = sum(
        result["similarity"]
        for result in results
    ) / total

    print("\n" + "=" * 60)
    print("RAG EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Total tests: {total}")
    print(f"Successful retrievals: {successful}")
    print(f"Accuracy: {accuracy:.2f}%")

    print(
        f"Average similarity: "
        f"{average_similarity:.4f}"
    )


def run_evaluation():
    url_tests = [
        {
            "query": (
                "Quote the sentence describing the "
                "purpose of Black Tech History Month."
            ),
            "expected_answer_snippet": (
                "PreparationTech is devoting our "
                "video interview platform"
            )
        }
    ]

    pdf_tests = [
        {
            "query": (
                "Retrieve the paragraph where recurrent "
                "models perform computations sequentially."
            ),
            "expected_answer_snippet": (
                "inherently sequential nature "
                "precludes parallelization"
            )
        }
    ]

    print("\nEvaluating URL collection...")

    url_results = evaluate_collection(
        URL_COLLECTION,
        url_tests
    )

    print("\nEvaluating PDF collection...")

    pdf_results = evaluate_collection(
        PDF_COLLECTION,
        pdf_tests
    )

    print_summary(
        url_results + pdf_results
    )


if __name__ == "__main__":
    run_evaluation()
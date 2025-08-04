import asyncio
import os
import random
import shutil
import time
from typing import List

import numpy as np
import torch
from langchain_core.documents import Document

from src.retriever.vector_store import AsyncVectorStore


def set_deterministic_seeds():
    """Set all random seeds for reproducible results."""
    random.seed(42)

    np.random.seed(42)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def cleanup_test_data():
    """Clean up test data directories."""
    test_dirs = ["./test_chroma_db"]

    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            print(f"🧹 Cleaning up {dir_path}")
            shutil.rmtree(dir_path)


def create_test_documents() -> List[Document]:
    """Create sample documents for testing with consistent content."""
    documents = [
        Document(
            page_content="""
            Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
            These machines can perform tasks that typically require human intelligence, such as visual perception,
            speech recognition, decision-making, and language translation. AI has applications in various fields
            including healthcare, finance, transportation, and entertainment.
            """,
            metadata={
                "source": "ai_intro.txt",
                "category": "technology",
                "title": "Introduction to AI",
                "test_id": "doc_001",
            },
        ),
        Document(
            page_content="""
            Machine Learning is a subset of AI that enables computers to learn and improve from experience
            without being explicitly programmed. It uses algorithms to identify patterns in data and make
            predictions or decisions. Common types include supervised learning, unsupervised learning, and
            reinforcement learning. Popular frameworks include TensorFlow, PyTorch, and scikit-learn.
            """,
            metadata={
                "source": "ml_basics.txt",
                "category": "technology",
                "title": "Machine Learning Basics",
                "test_id": "doc_002",
            },
        ),
        Document(
            page_content="""
            Deep Learning is a subset of machine learning that uses neural networks with multiple layers
            to model and understand complex patterns. It has revolutionized fields like computer vision,
            natural language processing, and speech recognition. Deep learning models can automatically
            learn hierarchical representations of data, making them powerful for complex tasks.
            """,
            metadata={
                "source": "deep_learning.txt",
                "category": "technology",
                "title": "Deep Learning Overview",
                "test_id": "doc_003",
            },
        ),
        Document(
            page_content="""
            Natural Language Processing (NLP) is a field of AI that focuses on the interaction between
            computers and human language. It enables machines to understand, interpret, and generate
            human language. Applications include chatbots, language translation, sentiment analysis,
            and text summarization. Modern NLP uses transformer models like BERT and GPT.
            """,
            metadata={
                "source": "nlp_intro.txt",
                "category": "technology",
                "title": "NLP Fundamentals",
                "test_id": "doc_004",
            },
        ),
        Document(
            page_content="""
            Computer Vision is a field of AI that enables computers to interpret and understand visual
            information from the world. It involves techniques for acquiring, processing, analyzing,
            and understanding digital images. Applications include facial recognition, autonomous vehicles,
            medical imaging, and augmented reality. Deep learning has significantly advanced this field.
            """,
            metadata={
                "source": "computer_vision.txt",
                "category": "technology",
                "title": "Computer Vision Basics",
                "test_id": "doc_005",
            },
        ),
    ]
    return documents


async def test_document_addition(vector_store: AsyncVectorStore) -> dict:
    """Test adding documents to the vector store with consistent results."""
    print("🧪 Testing document addition...")

    documents = create_test_documents()
    start_time = time.time()

    await vector_store.add_documents(documents)

    end_time = time.time()
    duration = end_time - start_time

    print(f"✅ Added {len(documents)} documents in {duration:.2f} seconds")

    return {
        "documents_added": len(documents),
        "duration": duration,
    }


async def test_search_functionality(vector_store: AsyncVectorStore) -> dict:
    """Test search functionality with consistent queries and expected results."""
    print("\n🔍 Testing search functionality...")

    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain deep learning neural networks",
        "Natural language processing applications",
        "Computer vision and image recognition",
    ]

    results_summary = {}

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        start_time = time.time()

        results = await vector_store.search(query, top_k=3)

        end_time = time.time()
        duration = end_time - start_time

        print(f"Search completed in {duration:.3f} seconds")

        query_results = []
        for i, (doc, score) in enumerate(results, 1):
            result_info = {
                "rank": i,
                "score": round(score, 4),
                "title": doc.metadata.get("title", "N/A"),
                "test_id": doc.metadata.get("test_id", "N/A"),
            }
            query_results.append(result_info)
            print(f"  {i}. Score: {score:.3f} | Title: {result_info['title']}")
            print(f"     Content: {doc.page_content[:100]}...")

        results_summary[query] = {"duration": duration, "results": query_results}

    return results_summary


async def test_batch_processing(vector_store: AsyncVectorStore) -> dict:
    """Test batch processing with consistent document sets."""
    print("\n📦 Testing batch processing...")

    # Create consistent batch of documents
    large_documents = []
    for i in range(20):
        doc = Document(
            page_content=f"""
            This is test document number {i + 1}. It contains information about various topics
            including technology, science, and innovation. Document {i + 1} discusses important
            concepts and provides valuable insights for testing purposes. The content includes
            multiple paragraphs with different types of information to test chunking and
            retrieval capabilities of the vector store system.
            """,
            metadata={
                "source": f"test_doc_{i + 1:03d}.txt",
                "batch_id": "test_batch_001",
                "index": i,
                "test_id": f"batch_doc_{i + 1:03d}",
            },
        )
        large_documents.append(doc)

    start_time = time.time()
    await vector_store.add_documents(large_documents)
    end_time = time.time()

    duration = end_time - start_time
    print(
        f"✅ Added {len(large_documents)} documents in batch in {duration:.2f} seconds"
    )

    return {
        "documents_added": len(large_documents),
        "duration": duration,
        "batch_id": "test_batch_001",
    }


async def test_performance(vector_store: AsyncVectorStore) -> dict:
    """Test performance with consistent batch sizes and measurements."""
    print("\n⚡ Testing performance...")

    # Test with different batch sizes
    batch_sizes = [4, 8, 16, 32, 64, 128]
    performance_results = {}

    for batch_size in batch_sizes:
        print(f"\nTesting with batch_size: {batch_size}")

        # Create a NEW vector store instance with this batch size
        test_config = {
            "collection_name": f"test_collection_batch_{batch_size}",
            "persist_directory": f"./test_chroma_db_batch_{batch_size}",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "batch_size": batch_size,  # This is the key difference
            "max_workers": 2,
            "top_k": 3,
            "recursive_character_text_splitter": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "length_function": len,
                "separators": ["\n\n", "\n", " ", ""],
            },
        }

        test_vector_store = AsyncVectorStore(test_config)

        # Create longer test documents that will be chunked
        test_docs = []
        for i in range(10):
            long_content = f"""
            Performance test document {i + 1:03d} with batch size {batch_size}. 
            This is a much longer document that will actually be processed by the text splitter.
            It contains multiple paragraphs of content to ensure that the chunking process
            creates multiple chunks from each document. This allows us to properly test
            the batching behavior of the vector store operations.
            
            The document continues with additional content to make it long enough for chunking.
            We need enough text so that when the RecursiveCharacterTextSplitter processes it,
            it creates multiple chunks instead of just one. This is important for testing
            the batch processing functionality of our vector store implementation.
            
            More content follows to ensure proper chunking. The chunk size is set to 500 characters,
            so we need documents that are significantly longer than that to see multiple chunks.
            This will help us understand how the batching affects performance when there are
            actually multiple chunks being processed in parallel.
            
            Additional paragraphs are added to make this document long enough. The goal is to
            have each document split into at least 3-4 chunks, so we can see how different
            batch sizes affect the processing time and throughput of the vector store operations.
            """.strip()

            doc = Document(
                page_content=long_content,
                metadata={
                    "batch_size": batch_size,
                    "doc_id": i,
                    "test_id": f"perf_doc_{batch_size}_{i + 1:03d}",
                },
            )
            test_docs.append(doc)

        print(f"  Created {len(test_docs)} long documents for testing")

        start_time = time.time()
        await test_vector_store.add_documents(test_docs)
        end_time = time.time()

        duration = end_time - start_time
        rate = len(test_docs) / duration if duration > 0 else 0

        print(f"  Time: {duration:.2f} seconds")
        print(f"  Rate: {rate:.1f} docs/second")

        performance_results[batch_size] = {
            "duration": duration,
            "rate": rate,
            "documents": len(test_docs),
        }

        # Clean up the test directory
        test_dir = f"./test_chroma_db_batch_{batch_size}"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    return performance_results


def verify_consistency(results: dict, expected_results: dict = None) -> bool:
    """Verify that results are consistent with expected values."""
    print("\n🔍 Verifying consistency...")

    is_consistent = True

    if "search_functionality" in results:
        search_results = results["search_functionality"]
        for query, query_data in search_results.items():
            if len(query_data["results"]) != 3:  # top_k=3
                print(
                    f"❌ Inconsistent search results for '{query}': {len(query_data['results'])} results"
                )
                is_consistent = False
            else:
                print(
                    f"✅ Search results consistent for '{query}': {len(query_data['results'])} results"
                )

    if "performance" in results:
        perf_results = results["performance"]
        for batch_size, perf_data in perf_results.items():
            if perf_data["documents"] != 10:
                print(
                    f"❌ Inconsistent performance test: {perf_data['documents']} docs (expected 10)"
                )
                is_consistent = False
            else:
                print(f"✅ Performance test consistent for batch_size {batch_size}")

            # Check that performance varies with batch size (should see different rates)
            print(f"   Batch size {batch_size}: {perf_data['rate']:.1f} docs/second")

    return is_consistent


async def main():
    """Main test function with consistency controls."""
    print("🚀 Starting Consistent AsyncVectorStore Test Driver")
    print("=" * 60)

    # Set deterministic seeds
    set_deterministic_seeds()
    print("🎲 Set deterministic random seeds")

    # Clean up any existing test data
    cleanup_test_data()

    # Create vector store with consistent configuration
    config = {
        "collection_name": "test_collection",
        "persist_directory": "./test_chroma_db",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "batch_size": 32,
        "max_workers": 2,
        "top_k": 3,
        "recursive_character_text_splitter": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "length_function": len,
            "separators": ["\n\n", "\n", " ", ""],
        },
    }

    vector_store = AsyncVectorStore(config)

    all_results = {}

    try:
        # Run tests and collect results
        all_results["document_addition"] = await test_document_addition(vector_store)
        all_results["search_functionality"] = await test_search_functionality(
            vector_store
        )
        all_results["batch_processing"] = await test_batch_processing(vector_store)
        all_results["performance"] = await test_performance(vector_store)

        # Verify consistency
        is_consistent = verify_consistency(all_results)

        if is_consistent:
            print("\n🎉 All tests completed successfully with consistent results!")
        else:
            print("\n⚠️ Tests completed but some inconsistencies detected!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        cleanup_test_data()
        print("\n🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())

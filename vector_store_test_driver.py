import asyncio
import time
from typing import List
from langchain_core.documents import Document

from src.retriever.vector_store import AsyncVectorStore


async def create_test_documents() -> List[Document]:
    """Create sample documents for testing."""
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
            },
        ),
    ]
    return documents


async def test_document_addition(vector_store: AsyncVectorStore) -> None:
    """Test adding documents to the vector store."""
    print("🧪 Testing document addition...")

    documents = await create_test_documents()
    start_time = time.time()

    await vector_store.add_documents(documents)

    end_time = time.time()
    print(f"✅ Added {len(documents)} documents in {end_time - start_time:.2f} seconds")


async def test_search_functionality(vector_store: AsyncVectorStore) -> None:
    """Test search functionality with various queries."""
    print("\n🔍 Testing search functionality...")

    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain deep learning neural networks",
        "Natural language processing applications",
        "Computer vision and image recognition",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        start_time = time.time()

        results = await vector_store.search(query, top_k=3)

        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.3f} seconds")

        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.3f} | Title: {doc.metadata.get('title', 'N/A')}")
            print(f"     Content: {doc.page_content[:100]}...")


async def test_batch_processing(vector_store: AsyncVectorStore) -> None:
    """Test batch processing with larger document sets."""
    print("\n📦 Testing batch processing...")

    # Create more documents for batch testing
    large_documents = []
    for i in range(20):
        doc = Document(
            page_content=f"""
            This is test document number {i+1}. It contains information about various topics
            including technology, science, and innovation. Document {i+1} discusses important
            concepts and provides valuable insights for testing purposes. The content includes
            multiple paragraphs with different types of information to test chunking and
            retrieval capabilities of the vector store system.
            """,
            metadata={"source": f"test_doc_{i+1}.txt", "batch_id": "test_batch", "index": i},
        )
        large_documents.append(doc)

    start_time = time.time()
    await vector_store.add_documents(large_documents)
    end_time = time.time()

    print(
        f"✅ Added {len(large_documents)} documents in batch in {end_time - start_time:.2f} seconds"
    )


async def test_performance(vector_store: AsyncVectorStore) -> None:
    """Test performance with different batch sizes."""
    print("\n⚡ Testing performance...")

    # Test with different batch sizes
    batch_sizes = [16, 32, 64, 128]

    for batch_size in batch_sizes:
        print(f"\nTesting with batch_size: {batch_size}")

        # Create test documents
        test_docs = []
        for i in range(50):
            doc = Document(
                page_content=f"Performance test document {i+1} with batch size {batch_size}.",
                metadata={"batch_size": batch_size, "doc_id": i},
            )
            test_docs.append(doc)

        start_time = time.time()
        await vector_store.add_documents(test_docs)
        end_time = time.time()

        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Rate: {len(test_docs)/(end_time - start_time):.1f} docs/second")


async def main():
    """Main test function."""
    print("🚀 Starting AsyncVectorStore Test Driver")
    print("=" * 50)

    # Create vector store with test configuration
    config = {
        "collection_name": "test_collection",
        "persist_directory": "./test_chroma_db",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "batch_size": 32,
        "max_workers": 2,
        "top_k": 3,
        "recursive_character_text_splitter": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "length_function": len,
            "separators": ["\n\n", "\n", " ", ""],
        },
    }

    vector_store = AsyncVectorStore(config)

    try:
        # Run tests
        await test_document_addition(vector_store)
        await test_search_functionality(vector_store)
        await test_batch_processing(vector_store)
        await test_performance(vector_store)

        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        await vector_store.close()
        print("\n🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())

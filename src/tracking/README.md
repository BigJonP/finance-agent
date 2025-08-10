# MLflow Tracking for Finance Agent

This module provides comprehensive MLflow tracking integration for the finance agent system, allowing you to monitor and analyze the performance of your document processing, search operations, and financial advice generation.

## Components

### 1. Retriever Tracking
Comprehensive tracking for document processing and vector search operations.

### 2. Advisor Tracking
Advanced tracking for financial advice generation, including prompt analysis and portfolio context processing.

## Features

- **Automatic Configuration Logging**: Vector store, embedder, and advisor configurations are automatically logged
- **Document Processing Metrics**: Track document chunking, processing time, and chunk size distributions
- **Search Performance Monitoring**: Monitor search query performance, response times, and result quality
- **Financial Advice Tracking**: Monitor prompt generation, advice quality, and response analysis
- **Portfolio Context Tracking**: Track user portfolio analysis and relevant document retrieval
- **Artifact Storage**: Store vector store artifacts, sample documents, configuration files, and prompts
- **Error Tracking**: Log and monitor errors that occur during processing
- **Flexible Configuration**: Environment variable overrides for MLflow settings

## Quick Start

### 1. Basic Usage

The tracking is automatically integrated into the vector store and advisor operations:

```python
from retriever.vector_store import get_vector_store
from retriever.config import VECTOR_STORE_CONFIG
from api.services.advisor import get_advisor

# Initialize vector store (configs are automatically logged)
vector_store = get_vector_store(VECTOR_STORE_CONFIG)

# Initialize advisor (configs and operations are automatically tracked)
advisor = get_advisor()

# Generate advice (all operations are automatically tracked)
advice = await advisor.generate_financial_advice(holdings, documents)
```

### 2. Manual Tracking

For custom tracking scenarios:

```python
from tracking import RetrieverTracker, AdvisorTracker

# Retriever tracking
with RetrieverTracker() as tracker:
    tracker.start_run(run_name="custom-retriever-experiment")
    # Your custom logic here

# Advisor tracking
with AdvisorTracker() as tracker:
    tracker.start_run(run_name="custom-advisor-experiment")
    # Your custom logic here
```

## Configuration

### Environment Variables

You can override default MLflow settings using environment variables:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_EXPERIMENT_NAME="my-custom-experiment"
export MLFLOW_ARTIFACT_LOCATION="./custom_artifacts"
```

### Default Configuration

- **Tracking URI**: `sqlite:///mlflow.db` (local SQLite database)
- **Experiment Name**: `finance-agent`
- **Artifact Location**: `./mlflow_artifacts`

## What Gets Tracked

### 1. Vector Store Configuration
- Collection name
- Chunk size and overlap
- Batch size
- Top-k parameter
- Full configuration JSON

### 2. Embedder Configuration
- Model name
- Device (CPU/GPU)
- Batch size
- Normalization settings
- Full configuration JSON

### 3. Document Processing Metrics
- Total documents processed
- Total chunks created
- Processing time
- Chunk size distribution (histogram)
- Sample documents (content and metadata)

### 4. Search Performance
- Query text and length
- Search time
- Number of results
- Score distributions
- Top-k parameter used

### 5. Financial Advisor Configuration
- Model name and parameters
- Retriever settings
- Token limits and temperature
- System prompt content
- Full configuration JSON

### 6. Portfolio Context Processing
- User ID and holdings count
- Relevant documents count and length
- Processing time for portfolio analysis
- Holdings summary and metadata
- Sample relevant documents

### 7. Advice Generation
- User prompt content and analysis
- Advice response content and analysis
- Generation time
- Token count (when available)
- Prompt quality metrics

### 8. Vector Store Search (Advisor)
- Stock-specific search queries
- Enhanced query content
- Search result scores and counts
- Search performance metrics

### 9. Error Logging
- Error type and message
- Context information
- User ID (when applicable)
- Timestamp

## Viewing Results

### 1. MLflow UI

Start the MLflow UI to view your experiments:

```bash
./start_mlflow_ui.sh
```

Then open your browser to `http://localhost:5000`

### 2. Programmatic Access

```python
from mlflow.tracking import MlflowClient

client = MlflowClient("sqlite:///mlflow.db")
experiments = client.list_experiments()
runs = client.search_runs(experiment_ids=[experiments[0].experiment_id])
```

## Best Practices

1. **Use Context Managers**: Always use the `with` statement for automatic run management
2. **Handle Errors Gracefully**: The tracking system won't fail your main operations
3. **Monitor Artifact Storage**: Ensure you have sufficient disk space for artifacts
4. **Regular Cleanup**: Periodically clean up old MLflow runs and artifacts
5. **Environment Isolation**: Use different experiment names for different environments
6. **Prompt Analysis**: Review logged prompts to improve advice quality
7. **Performance Monitoring**: Track generation times to optimize response speed

## Troubleshooting

### Common Issues

1. **MLflow Import Error**: Ensure `mlflow` is in your requirements.txt
2. **Permission Errors**: Check write permissions for artifact directories
3. **Database Locked**: Ensure only one process accesses the SQLite database
4. **Artifact Storage Full**: Monitor disk space and clean up old artifacts
5. **Prompt Logging Issues**: Check if prompts contain sensitive information

### Debug Mode

Enable debug logging by setting the environment variable:

```bash
export MLFLOW_TRACKING_DEBUG=1
```

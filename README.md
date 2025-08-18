# 🤖 Finance Agent - AI-Powered Financial Advisory Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Intelligent Financial Advisory System** - Leveraging AI, Machine Learning, and Vector Search to provide personalized investment advice and portfolio management solutions.

**Frontend UI**: A modern Next.js TypeScript application is available at [finance-agent-ui](https://github.com/BigJonP/finance-agent-ui) for the complete user experience.

## 🎥 Demo Video

[![Demo Video](https://img.youtube.com/vi/QMQq0d4y19c/0.jpg)](https://youtu.be/QMQq0d4y19c)

*Click the thumbnail above to watch the demo video on YouTube!*

## 🚀 What is Finance Agent?

**Finance Agent** is an AI-powered financial advisory platform that combines large language models, vector search, and reddit to deliver personalized investment advice and portfolio management tips. Built with modern Python technologies, it provides a robust API for financial applications, investment platforms, and personal finance tools.

### ✨ Key Features

- **🤖 AI-Powered Financial Advice** - Generate personalized investment recommendations using LLMs
- **📊 Portfolio Management** - Track and manage investment holdings with real-time updates (reddit posts)
- **🔍 Intelligent Document Search** - Vector-based search through financial documents and reports
- **🔐 Secure Authentication** - JWT-based user authentication and authorization
- **📈 MLflow Integration** - Comprehensive machine learning experiment tracking
- **🚀 FastAPI Backend** - High-performance, async API
- **💾 Vector Database** - ChromaDB integration for semantic search capabilities
- **🌐 Modern Web API** - RESTful endpoints with CORS support and comprehensive error handling

## 🏗️ Architecture & Technology Stack

### Backend Technologies
- **FastAPI** - Modern, fast web framework for building APIs
- **Python 3.8+** - Core programming language
- **Pydantic** - Data validation and settings management

### AI & Machine Learning
- **LangChain** - Framework for developing applications with LLMs
- **OpenAI GPT** - Large language model integration
- **Sentence Transformers** - Text embedding and similarity search
- **ChromaDB** - Vector database for semantic search
- **MLflow** - Machine learning lifecycle management

### Database & Storage
- **Supabase** - PostgreSQL-based backend
- **ChromaDB** - Vector store for document embeddings
- **Async Support** - High-performance database operations

### Security & Authentication
- **JWT Tokens** - Secure user authentication
- **bcrypt** - Password hashing and security
- **CORS Middleware** - Cross-origin resource sharing

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Supabase account and credentials

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/finance-agent.git
   cd finance-agent
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start the API**
   ```bash
   sh start_api.sh
   ```

### Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
LOG_LEVEL=INFO
```

## 📚 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Core Endpoints

- **`POST /generate_advice`** - Generate AI-powered financial advice
- **`GET /holding/`** - Retrieve user investment holdings
- **`POST /holding/`** - Add new investment holding
- **`DELETE /holding/`** - Remove investment holding
- **`POST /auth/login`** - User authentication
- **`POST /auth/register`** - User registration

## 🔧 Development

### Project Structure
```
finance-agent/
├── src/
│   ├── api/             # FastAPI application and routes
│   ├── retriever/       # Vector search and document processing
│   ├── db/              # Database utilities and models
│   └── tracking/        # MLflow integration
├── assets/              # Demo videos and static files
├── tests/               # Test suite
├── requirements.txt     # Python dependencies
└── start_api.sh         # Startup script
└── start_mlflow_ui.sh   # Launch mlflow UI
```

## 🎯 Use Cases

### For Individual Investors
- **Personalized Investment Advice** - Get AI-generated recommendations based on your portfolio
- **Financial Education** - Learn about investment strategies and market analysis

### For Fintech Applications
- **API Integration** - Integrate AI-powered financial advice into existing applications
- **Robo-Advisory Services** - Build automated investment advisory platforms
- **Financial Chatbots** - Create intelligent financial assistance bots

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.


## Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/finance-agent/issues)
- **Discussions**: [Join the community](https://github.com/yourusername/finance-agent/discussions)
- **Author**: [Jonathan Paserman](https://github.com/BigJonP)

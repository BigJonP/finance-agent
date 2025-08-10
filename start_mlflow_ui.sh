echo "🚀 Starting MLflow UI..."
echo "📊 Tracking URI: sqlite:///mlflow.db"
echo "📊 Experiment: finance-agent-retriever"
echo "🌐 UI will be available at: http://localhost:5000"
echo ""

if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not detected. Activating..."
    source venv/bin/activate
fi

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

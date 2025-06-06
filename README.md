# Intent Classification API - Prozis Challenge

🚀 **REST API for Portuguese text intent classification in e-commerce context**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![Machine Learning](https://img.shields.io/badge/ML-SGDClassifier-orange.svg)](https://scikit-learn.org)

## 📋 Overview

This API classifies Portuguese text into 6 predefined e-commerce intents using machine learning. **The model was trained on an enhanced dataset with 3 variations per labeled case of the original dataset for improved robustness.** Built with FastAPI and scikit-learn's SGDClassifier, it provides real-time text classification for customer interactions in an online supplement store context.

### 🎯 Supported Intents

| Intent | Description | Examples |
|--------|-------------|----------|
| `search_product` | Product search queries | "whey protein", "ver multivitamínicos", "barra de amendoim" |
| `product_info` | Information requests | "como tomar creatina", "benefícios omega 3", "diferença entre whey concentrada e isolada" |
| `add_to_cart` | Add items to cart | "adicionar ao carrinho", "quero 2 unidades", "põe a creatina no cesto" |
| `go_to_shopping_cart` | Navigate to cart | "ver carrinho", "mostrar cesto", "ir para o cesto" |
| `confirm_order` | Complete purchase | "finalizar compra", "pagar agora", "confirmar encomenda" |
| `unknown_intention` | Unclear or generic text | "obrigado", "preciso de ajuda", "promoções?" |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd intent-classification-api
```

2. **Install dependencies**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

3. **Prepare your data** (Optional)
   - Place your enhanced dataset at `data/enhanced_intent_dataset.json`
   - Or use the provided sample data (automatically generated)

4. **Start the server**
```bash
python main.py
```

The API will be available at:
- **Main API**: http://127.0.0.1:8000
- **Interactive Docs**: http://127.0.0.1:8000/docs
- **Alternative Docs**: http://127.0.0.1:8000/redoc

## 📚 API Usage

### Single Text Classification

**Endpoint**: `POST /api/v1/classify`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "whey protein isolada"}'
```

**Response**:
```json
{
  "intent": "search_product",
  "confidence_score": 0.9234,
  "processing_time_ms": 12.45
}
```

### Batch Classification

**Endpoint**: `POST /api/v1/classify/batch`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "whey protein",
      "como tomar creatina", 
      "adicionar ao carrinho"
    ]
  }'
```

**Response**:
```json
{
  "intent": [
    {
      "intent": "search_product",
      "confidence_score": 0.9019,
      "processing_time_ms": 7.24
    },
    {
      "intent": "product_info",
      "confidence_score": 0.8186,
      "processing_time_ms": 1.76
    },
    {
      "intent": "add_to_cart",
      "confidence_score": 0.505,
      "processing_time_ms": 1.46
    }
  ],
  "total_processed": 3,
  "avg_processing_time_ms": 3.72
```

### Health Check

**Endpoint**: `GET /api/v1/health`

```bash
curl "http://127.0.0.1:8000/api/v1/health"
```

### Model Information

**Endpoint**: `GET /api/v1/model/info`

```bash
curl "http://127.0.0.1:8000/api/v1/model/info"
```

### Retrain Model

**Endpoint**: `POST /api/v1/model/retrain`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/model/retrain"
```

### Usage Statistics

**Endpoint**: `GET /api/v1/stats`

```bash
curl "http://127.0.0.1:8000/api/v1/stats"
```

## 🔧 Configuration

### Environment Variables

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL=INFO
```

### Data Files Priority

The API looks for training data in this order:
1. `data/enhanced_intent_dataset.json` (Enhanced dataset)
2. `data/intent_dataset.json` (Original dataset)  
3. `data/intent_dataset.csv` (CSV format)
4. Auto-generated sample data (fallback)

## 🧪 Testing

### Run All Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run comprehensive test suite
python -m pytest tests/test_api.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/test_api.py --cov=. --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing  
- **E2E Tests**: End-to-end workflow testing
- **Performance Tests**: Response time validation
- **Validation Tests**: Input validation testing

### Example Tests

```bash
# Test single classification
python -m pytest tests/test_api.py::TestClassificationEndpoint::test_classify_text_valid_inputs -v

# Test batch processing
python -m pytest tests/test_api.py::TestBatchClassificationEndpoint::test_batch_classify_valid_inputs -v

# Test Portuguese language support
python -m pytest tests/test_api.py::TestIntegrationScenarios::test_portuguese_language_support -v
```

## 📊 Model Details

### Architecture
- **Algorithm**: SGDClassifier (Stochastic Gradient Descent)
- **Vectorizer**: TF-IDF with n-grams
- **Features**: 1000+ text features
- **Languages**: Portuguese (primary), English (secondary)

### Performance Metrics
- **Training Accuracy**: ~95%+ (with enhanced dataset)
- **Cross-validation**: 5-fold CV
- **Response Time**: <100ms average per classification
- **Batch Processing**: ~10ms average per text

### Model Training
```python
# The model automatically trains on startup if no saved model exists
# Training data format (JSON):
[
  {"text": "whey protein", "intent": "search_product"},
  {"text": "como tomar creatina", "intent": "product_info"},
  ...
]
```

The model chosen was SGDClassifier due to the results that came from the script app/utils/benchmark_models.py

```bash
python app\utils\benchmark_models.py
```

| Model                    | Mean Accuracy |
|--------------------------|----------------|
| LogisticRegression       | 0.8222         |
| RandomForest             | 0.5889         |
| ComplementNB             | 0.8222         |
| XGBoost                  | 0.4444         |
| SGDClassifier (tuned)    | 0.8444         |
| LinearSVC (tuned)        | 0.8444         |



## 🔮 Suggested Enhancements
Context-Aware Inference: Incorporate user session history or interaction flow for improved accuracy in ambiguous cases.

Model Deployment: Add Docker and CI/CD for production readiness.

Richer Dataset: Augment training data with real-world user queries or perform active learning with production inputs.

Intent Confidence Calibration: Improve reliability of confidence scores with techniques like Platt scaling.

Fallback Strategies: Use ensemble models or rule-based logic for low-confidence or unknown classifications.

Streaming & Async Support: Enable real-time input via websockets or Kafka integration.

Monitoring: Log and analyze user queries in production for model drift detection and retraining triggers.

Multilingual Support: Extend model to support more languages, given the e-commerce context might attract international users.
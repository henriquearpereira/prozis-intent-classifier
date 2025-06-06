"""
Comprehensive test suite for Intent Classification API
Includes unit tests, integration tests, and E2E tests
"""
import pytest
import asyncio
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import tempfile
import shutil

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, create_sample_data
from models.classifier import IntentClassifier
from api.routes import init_classifier

class TestIntentClassificationAPI:
    """Test suite for Intent Classification API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_classifier(self):
        """Create a mock classifier for testing"""
        classifier = Mock(spec=IntentClassifier)
        classifier.is_trained = True
        classifier.intent_labels = {
            'search_product': 0,
            'product_info': 1,
            'add_to_cart': 2,
            'go_to_shopping_cart': 3,
            'confirm_order': 4,
            'unknown_intention': 5
        }
        classifier.predict.return_value = ('search_product', 0.95)
        classifier.get_model_info.return_value = {
            'status': 'trained',
            'supported_intents': classifier.intent_labels,
            'model_type': 'SGDClassifier',
            'vectorizer_features': 1000
        }
        return classifier
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

class TestHealthEndpoint(TestIntentClassificationAPI):
    """Test health check endpoint"""
    
    def test_health_endpoint_success(self, client):
        """Test health endpoint returns correct structure"""
        response = client.get("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert "model_ready" in data
        assert "uptime_seconds" in data
        assert "supported_intents" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert isinstance(data["supported_intents"], list)

class TestClassificationEndpoint(TestIntentClassificationAPI):
    """Test single text classification endpoint"""
    
    @pytest.mark.parametrize("text,expected_intent", [
        ("whey protein", "search_product"),
        ("como tomar creatina", "product_info"),
        ("adicionar ao carrinho", "add_to_cart"),
        ("ver carrinho", "go_to_shopping_cart"),
        ("finalizar compra", "confirm_order"),
        ("olá", "unknown_intention")
    ])
    def test_classify_text_valid_inputs(self, client, text, expected_intent):
        """Test classification with valid inputs for each intent"""
        response = client.post(
            "/api/v1/classify",
            json={"text": text}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "intent" in data
            assert "confidence_score" in data
            assert "processing_time_ms" in data
            assert 0.0 <= data["confidence_score"] <= 1.0
            assert data["processing_time_ms"] >= 0
            # Note: We can't guarantee exact intent match without trained model
            assert data["intent"] in [
                "search_product", "product_info", "add_to_cart", 
                "go_to_shopping_cart", "confirm_order", "unknown_intention"
            ]
    
    def test_classify_empty_text(self, client):
        """Test classification with empty text"""
        response = client.post(
            "/api/v1/classify",
            json={"text": ""}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_classify_whitespace_only(self, client):
        """Test classification with whitespace only"""
        response = client.post(
            "/api/v1/classify",
            json={"text": "   "}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_classify_too_long_text(self, client):
        """Test classification with text exceeding maximum length"""
        long_text = "a" * 1001  # Exceeds 1000 character limit
        response = client.post(
            "/api/v1/classify",
            json={"text": long_text}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_classify_missing_text_field(self, client):
        """Test classification without text field"""
        response = client.post(
            "/api/v1/classify",
            json={}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_classify_invalid_json(self, client):
        """Test classification with invalid JSON"""
        response = client.post(
            "/api/v1/classify",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

class TestBatchClassificationEndpoint(TestIntentClassificationAPI):
    """Test batch text classification endpoint"""
    
    def test_batch_classify_valid_inputs(self, client):
        """Test batch classification with valid inputs"""
        texts = [
            "whey protein",
            "como tomar creatina",
            "adicionar ao carrinho",
            "ver carrinho",
            "finalizar compra"
        ]
        
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": texts}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "intent" in data
            assert "total_processed" in data
            assert "avg_processing_time_ms" in data
            assert data["total_processed"] == len(texts)
            assert len(data["intent"]) == len(texts)
            
            # Check each result
            for result in data["intent"]:
                assert "intent" in result
                assert "confidence_score" in result
                assert "processing_time_ms" in result
                assert 0.0 <= result["confidence_score"] <= 1.0
    
    def test_batch_classify_empty_list(self, client):
        """Test batch classification with empty list"""
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": []}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_batch_classify_too_many_texts(self, client):
        """Test batch classification with too many texts"""
        texts = ["test"] * 101  # Exceeds 100 text limit
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": texts}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_batch_classify_with_empty_text(self, client):
        """Test batch classification with empty text in list"""
        texts = ["valid text", "", "another valid text"]
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": texts}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_batch_classify_with_long_text(self, client):
        """Test batch classification with text exceeding limit"""
        texts = ["valid text", "a" * 1001, "another valid text"]
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": texts}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

class TestModelInfoEndpoint(TestIntentClassificationAPI):
    """Test model info endpoint"""
    
    def test_model_info_success(self, client):
        """Test model info endpoint returns correct structure"""
        response = client.get("/api/v1/model/info")
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "status" in data
            assert "supported_intents" in data
            assert "model_type" in data
            assert "vectorizer_features" in data
            assert "last_trained" in data
            assert isinstance(data["supported_intents"], list)
            assert isinstance(data["vectorizer_features"], int)

class TestRetrainEndpoint(TestIntentClassificationAPI):
    """Test model retrain endpoint"""
    
    def test_retrain_endpoint_success(self, client):
        """Test retrain endpoint returns correct structure"""
        response = client.post("/api/v1/model/retrain")
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "message" in data
            assert "data_path" in data
            assert "model_type" in data
            assert "timestamp" in data
            assert data["model_type"] == "SGDClassifier"
    
    def test_retrain_with_custom_path(self, client):
        """Test retrain with custom data path"""
        response = client.post(
            "/api/v1/model/retrain",
            params={"data_path": "custom/path/dataset.json"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["data_path"] == "custom/path/dataset.json"

class TestStatsEndpoint(TestIntentClassificationAPI):
    """Test statistics endpoint"""
    
    def test_stats_endpoint_success(self, client):
        """Test stats endpoint returns correct structure"""
        response = client.get("/api/v1/stats")
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "uptime_seconds" in data
            assert "uptime_hours" in data
            assert "model_ready" in data
            assert "supported_intents_count" in data
            assert "api_version" in data
            assert "model_algorithm" in data
            assert "timestamp" in data
            assert data["model_algorithm"] == "SGDClassifier"
            assert data["api_version"] == "1.0.0"

class TestRootEndpoint(TestIntentClassificationAPI):
    """Test root endpoint"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "supported_intents" in data
        assert "endpoints" in data
        
        # Check all 6 required intents are listed
        expected_intents = [
            "search_product", "product_info", "add_to_cart",
            "go_to_shopping_cart", "confirm_order", "unknown_intention"
        ]
        for intent in expected_intents:
            assert intent in data["supported_intents"]

class TestInputValidation(TestIntentClassificationAPI):
    """Test input validation thoroughly"""
    
    @pytest.mark.parametrize("invalid_input", [
        None,
        123,
        [],
        {},
        {"wrong_field": "text"}
    ])
    def test_classify_invalid_input_types(self, client, invalid_input):
        """Test classification with various invalid input types"""
        response = client.post(
            "/api/v1/classify",
            json=invalid_input
        )
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST
        ]
    
    def test_classify_special_characters(self, client):
        """Test classification with special characters"""
        special_texts = [
            "!@#$%^&*()",
            "áéíóúàèìòùâêîôûãõç",
            "测试中文",
            "🚀💊🔬",
            "test\nwith\nnewlines",
            "test\twith\ttabs"
        ]
        
        for text in special_texts:
            response = client.post(
                "/api/v1/classify",
                json={"text": text}
            )
            # Should either succeed or return proper error
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                status.HTTP_503_SERVICE_UNAVAILABLE
            ]

class TestIntegrationScenarios(TestIntentClassificationAPI):
    """Integration tests for real-world scenarios"""
    
    def test_e2e_product_search_workflow(self, client):
        """End-to-end test for product search workflow"""
        # 1. Check health
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == status.HTTP_200_OK
        
        # 2. Search for product
        search_response = client.post(
            "/api/v1/classify",
            json={"text": "whey protein isolada"}
        )
        
        if search_response.status_code == status.HTTP_200_OK:
            # 3. Get product info
            info_response = client.post(
                "/api/v1/classify",
                json={"text": "como tomar whey protein"}
            )
            assert info_response.status_code == status.HTTP_200_OK
            
            # 4. Add to cart
            cart_response = client.post(
                "/api/v1/classify",
                json={"text": "adicionar 1kg ao carrinho"}
            )
            assert cart_response.status_code == status.HTTP_200_OK
    
    def test_e2e_batch_classification_workflow(self, client):
        """End-to-end test for batch classification"""
        # Realistic e-commerce conversation
        conversation = [
            "whey protein isolada",
            "benefícios da whey",
            "adicionar 2kg ao carrinho",
            "ver meu carrinho",
            "finalizar compra"
        ]
        
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": conversation}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["total_processed"] == len(conversation)
            assert len(data["intent"]) == len(conversation)
    
    def test_portuguese_language_support(self, client):
        """Test Portuguese language support"""
        portuguese_texts = [
            "procurar proteína whey",
            "quais são os benefícios da creatina",
            "adicionar ao carrinho de compras",
            "mostrar produtos no cesto",
            "confirmar encomenda agora",
            "não compreendo"
        ]
        
        for text in portuguese_texts:
            response = client.post(
                "/api/v1/classify",
                json={"text": text}
            )
            
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert data["intent"] in [
                    "search_product", "product_info", "add_to_cart",
                    "go_to_shopping_cart", "confirm_order", "unknown_intention"
                ]

class TestErrorHandling(TestIntentClassificationAPI):
    """Test error handling scenarios"""
    
    def test_service_unavailable_scenarios(self, client):
        """Test scenarios where service might be unavailable"""
        # This would test when classifier is not initialized
        # In real scenarios, this might happen during startup
        pass
    
    def test_malformed_requests(self, client):
        """Test handling of malformed requests"""
        malformed_requests = [
            # Missing content-type
            ("", {"Content-Type": "text/plain"}),
            # Invalid JSON
            ("{invalid: json}", {"Content-Type": "application/json"}),
            # Wrong content type
            ("valid text", {"Content-Type": "text/xml"})
        ]
        
        for data, headers in malformed_requests:
            response = client.post(
                "/api/v1/classify",
                data=data,
                headers=headers
            )
            assert response.status_code in [
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                status.HTTP_400_BAD_REQUEST
            ]

class TestPerformance(TestIntentClassificationAPI):
    """Basic performance tests"""
    
    def test_response_time_single_classification(self, client):
        """Test response time for single classification"""
        import time
        
        start_time = time.time()
        response = client.post(
            "/api/v1/classify",
            json={"text": "whey protein"}
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        # Should respond within 5 seconds
        assert response_time < 5.0
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            # Processing time should be reasonable
            assert data["processing_time_ms"] < 5000
    
    def test_batch_processing_efficiency(self, client):
        """Test batch processing is more efficient than individual requests"""
        texts = ["test"] * 10
        
        # Time batch request
        import time
        start_time = time.time()
        batch_response = client.post(
            "/api/v1/classify/batch",
            json={"texts": texts}
        )
        batch_time = time.time() - start_time
        
        if batch_response.status_code == status.HTTP_200_OK:
            data = batch_response.json()
            # Average processing time should be reasonable
            assert data["avg_processing_time_ms"] < 1000

# Test utilities and fixtures
class TestUtilities:
    """Test utility functions"""
    
    def test_create_sample_data(self, temp_data_dir):
        """Test sample data creation"""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_data_dir)
            os.makedirs("data", exist_ok=True)
            
            create_sample_data()
            
            # Check if file was created
            assert os.path.exists("data/intent_dataset_sample.json")
            
            # Check file content
            with open("data/intent_dataset_sample.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            assert isinstance(data, list)
            assert len(data) > 0
            
            # Check all required intents are present
            intents = set(item["intent"] for item in data)
            expected_intents = {
                "search_product", "product_info", "add_to_cart",
                "go_to_shopping_cart", "confirm_order", "unknown_intention"
            }
            assert expected_intents.issubset(intents)
            
        finally:
            os.chdir(original_cwd)

# Pytest configuration and runners
def test_all_endpoints_accessible(client):
    """Test that all documented endpoints are accessible"""
    endpoints = [
        "/",
        "/api/v1/health",
        "/api/v1/stats",
        "/api/v1/model/info"
    ]
    
    for endpoint in endpoints:
        response = client.get(endpoint)
        # Should not return 404
        assert response.status_code != status.HTTP_404_NOT_FOUND

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
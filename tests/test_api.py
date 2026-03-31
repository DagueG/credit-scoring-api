"""Unit tests for credit scoring API."""

import time
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Fixture for FastAPI test client.
    
    Yields:
        TestClient: Client for making requests to the app
    """
    return TestClient(app)


@pytest.fixture
def valid_payload() -> dict:
    """Fixture for valid credit scoring request payload.
    
    Returns:
        dict: Valid input features for credit scoring
    """
    return {
        "EXT_SOURCE_1": 0.5019415614426349,
        "EXT_SOURCE_2": 0.3662832999999999,
        "EXT_SOURCE_3": 0.7119315,
        "AMT_CREDIT": 179055.0,
        "AMT_INCOME_TOTAL": 112500.0,
        "AMT_ANNUITY": 3951.0,
        "DAYS_BIRTH": -13363,
        "DAYS_EMPLOYED": -640,
    }


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health(self, client: TestClient) -> None:
        """Test health endpoint returns 200 and expected response.
        
        Args:
            client: FastAPI test client
        """
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""
    
    def test_predict_valid(self, client: TestClient, valid_payload: dict) -> None:
        """Test predict endpoint with valid input returns correct structure.
        
        Args:
            client: FastAPI test client
            valid_payload: Valid request payload
        """
        response = client.post("/predict", json=valid_payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that score is between 0 and 1
        assert 0.0 <= data["score"] <= 1.0
        
        # Check that prediction is 0 or 1
        assert data["prediction"] in [0, 1]
        
        # Check that threshold is present
        assert "threshold" in data
        assert isinstance(data["threshold"], float)
    
    def test_predict_response_format(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Test that predict response contains all expected fields.
        
        Args:
            client: FastAPI test client
            valid_payload: Valid request payload
        """
        response = client.post("/predict", json=valid_payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        required_fields = {"client_id", "score", "prediction", "threshold"}
        assert required_fields.issubset(data.keys())
        
        # Verify field types
        assert isinstance(data["client_id"], str)
        assert isinstance(data["score"], float)
        assert isinstance(data["prediction"], int)
        assert isinstance(data["threshold"], float)


class TestInputValidation:
    """Tests for input validation and error handling."""
    
    def test_predict_missing_fields(self, client: TestClient, valid_payload: dict) -> None:
        """Test that missing required fields returns 422.
        
        Args:
            client: FastAPI test client
            valid_payload: Valid request payload
        """
        # Remove a required field
        incomplete_payload = valid_payload.copy()
        del incomplete_payload["EXT_SOURCE_1"]
        
        response = client.post("/predict", json=incomplete_payload)
        assert response.status_code == 422
    
    def test_predict_wrong_types(self, client: TestClient, valid_payload: dict) -> None:
        """Test that wrong data types returns 422.
        
        Args:
            client: FastAPI test client
            valid_payload: Valid request payload
        """
        # Change a float field to string
        invalid_payload = valid_payload.copy()
        invalid_payload["EXT_SOURCE_1"] = "not_a_number"
        
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422
    
    def test_predict_empty_body(self, client: TestClient) -> None:
        """Test that empty body returns 422.
        
        Args:
            client: FastAPI test client
        """
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestBusinessLogic:
    """Tests for business logic and consistency."""
    
    def test_score_range(self, client: TestClient, valid_payload: dict) -> None:
        """Test that score is always between 0.0 and 1.0.
        
        Args:
            client: FastAPI test client
            valid_payload: Valid request payload
        """
        for _ in range(10):  # Make multiple requests to test randomness
            response = client.post("/predict", json=valid_payload)
            assert response.status_code == 200
            
            score = response.json()["score"]
            assert 0.0 <= score <= 1.0, f"Score {score} is out of range [0, 1]"
    
    def test_prediction_matches_threshold(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Test that prediction is consistent with score and threshold.
        
        prediction should be 1 if score >= threshold, else 0.
        
        Args:
            client: FastAPI test client
            valid_payload: Valid request payload
        """
        for _ in range(10):  # Make multiple requests
            response = client.post("/predict", json=valid_payload)
            assert response.status_code == 200
            
            data = response.json()
            score = data["score"]
            prediction = data["prediction"]
            threshold = data["threshold"]
            
            expected_prediction = 1 if score >= threshold else 0
            assert (
                prediction == expected_prediction
            ), f"Score {score}, threshold {threshold}, " \
               f"expected prediction {expected_prediction}, got {prediction}"


class TestPerformance:
    """Tests for performance and timing requirements."""
    
    def test_response_time(self, client: TestClient, valid_payload: dict) -> None:
        """Test that predict response completes in less than 500ms.
        
        Args:
            client: FastAPI test client
            valid_payload: Valid request payload
        """
        start_time = time.time()
        response = client.post("/predict", json=valid_payload)
        elapsed_time_ms = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert (
            elapsed_time_ms < 500
        ), f"Response took {elapsed_time_ms:.2f}ms, expected < 500ms"

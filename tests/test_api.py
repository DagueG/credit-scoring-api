"""Unit tests for credit scoring API."""

import json
import math
import os
import time

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.model import initialize_model_service


@pytest.fixture(scope="session", autouse=True)
def setup_model_service() -> None:
    """Initialize model service for all tests (session scope).
    
    This fixture runs once per test session before any tests.
    """
    initialize_model_service()


@pytest.fixture
def client() -> TestClient:
    """Fixture for FastAPI test client.
    
    Yields:
        TestClient: Client for making requests to the app
    """
    return TestClient(app)


@pytest.fixture
def valid_client_id() -> int:
    """Fixture for valid client ID from reference data.
    
    Returns:
        int: A valid SK_ID_CURR from the reference dataset (as Python int, not numpy int64)
    """
    try:
        # Load reference data to get a valid client ID
        df = pd.read_parquet('data/clients_reference.parquet')
        # Return the first client ID (index), converting to Python int
        return int(df.index[0])
    except Exception:
        # Fallback if reference data not available
        pytest.skip("Reference data not available for testing")


@pytest.fixture
def invalid_client_id() -> int:
    """Fixture for invalid client ID.
    
    Returns:
        int: A client ID that doesn't exist in reference data
    """
    return 999999999  # Unlikely to exist


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
    
    def test_predict_valid(
        self, client: TestClient, valid_client_id: int
    ) -> None:
        """Test predict endpoint with valid client returns correct structure.
        
        Args:
            client: FastAPI test client
            valid_client_id: Valid client ID for testing
        """
        response = client.post(
            "/predict",
            json={"client_id": valid_client_id}
        )
        
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
        self, client: TestClient, valid_client_id: int
    ) -> None:
        """Test that predict response contains all expected fields.
        
        Args:
            client: FastAPI test client
            valid_client_id: Valid client ID for testing
        """
        response = client.post(
            "/predict",
            json={"client_id": valid_client_id}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        required_fields = {
            "client_id", "score", "prediction", "threshold", "inference_time_ms"
        }
        assert required_fields.issubset(data.keys())
        
        # Verify field types
        assert isinstance(data["client_id"], int)
        assert isinstance(data["score"], float)
        assert isinstance(data["prediction"], int)
        assert isinstance(data["threshold"], float)
        assert isinstance(data["inference_time_ms"], float)


class TestInputValidation:
    """Tests for input validation and error handling."""
    
    def test_predict_missing_fields(self, client: TestClient) -> None:
        """Test that missing required fields returns 422.
        
        Args:
            client: FastAPI test client
        """
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    def test_predict_wrong_types(self, client: TestClient) -> None:
        """Test that wrong data types returns 422.
        
        Args:
            client: FastAPI test client
        """
        response = client.post(
            "/predict",
            json={"client_id": "not_an_int"}
        )
        assert response.status_code == 422
    
    def test_predict_empty_body(self, client: TestClient) -> None:
        """Test that empty body returns 422.
        
        Args:
            client: FastAPI test client
        """
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    def test_predict_unknown_client(
        self, client: TestClient, invalid_client_id: int
    ) -> None:
        """Test that unknown client_id returns 404.
        
        Args:
            client: FastAPI test client
            invalid_client_id: Invalid client ID
        """
        response = client.post(
            "/predict",
            json={"client_id": invalid_client_id}
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestBusinessLogic:
    """Tests for business logic and consistency."""
    
    def test_score_range(
        self, client: TestClient, valid_client_id: int
    ) -> None:
        """Test that score is always between 0.0 and 1.0.
        
        Args:
            client: FastAPI test client
            valid_client_id: Valid client ID for testing
        """
        response = client.post(
            "/predict",
            json={"client_id": valid_client_id}
        )
        assert response.status_code == 200
        
        score = response.json()["score"]
        assert 0.0 <= score <= 1.0, f"Score {score} is out of range [0, 1]"
    
    def test_prediction_matches_threshold(
        self, client: TestClient, valid_client_id: int
    ) -> None:
        """Test that prediction is consistent with score and threshold.
        
        prediction should be 1 if score >= threshold, else 0.
        
        Args:
            client: FastAPI test client
            valid_client_id: Valid client ID for testing
        """
        response = client.post(
            "/predict",
            json={"client_id": valid_client_id}
        )
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
    
    def test_response_time(
        self, client: TestClient, valid_client_id: int
    ) -> None:
        """Test that predict response completes in less than 500ms.
        
        Args:
            client: FastAPI test client
            valid_client_id: Valid client ID for testing
        """
        start_time = time.time()
        response = client.post(
            "/predict",
            json={"client_id": valid_client_id}
        )
        elapsed_time_ms = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert (
            elapsed_time_ms < 500
        ), f"Response took {elapsed_time_ms:.2f}ms, expected < 500ms"


class TestSampleClients:
    """Tests for sample client endpoint."""
    
    def test_sample_clients(self, client: TestClient) -> None:
        """Test that sample clients endpoint returns valid data.
        
        Args:
            client: FastAPI test client
        """
        response = client.get("/clients/sample?count=5")
        assert response.status_code == 200
        
        data = response.json()
        assert "client_ids" in data
        assert isinstance(data["client_ids"], list)
        assert len(data["client_ids"]) <= 5
        assert all(isinstance(cid, int) for cid in data["client_ids"])


class TestLogging:
    """Tests for structured JSON logging."""
    
    def test_logs_are_valid_json(
        self, client: TestClient, valid_client_id: int
    ) -> None:
        """Test that all logged predictions are valid JSON (no NaN values).
        
        Verifies:
        - Log file is created or appended to
        - Each logged line is valid JSON (strict parsing, no NaN/Infinity)
        - All required fields are present
        - Features don't contain NaN values
        
        Args:
            client: FastAPI test client
            valid_client_id: Valid client ID for testing
        """
        log_file = "logs/api_logs.jsonl"
        
        # Count initial log lines (file may have logs from previous tests)
        initial_line_count = 0
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                initial_line_count = len(f.readlines())
        
        # Make a prediction
        response = client.post(
            "/predict",
            json={"client_id": valid_client_id}
        )
        assert response.status_code == 200, "Prediction failed"
        
        # Read and verify logs exist
        assert os.path.exists(log_file), "Log file was not created"
        
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
        
        # Get only the newly added log lines
        new_log_lines = all_lines[initial_line_count:]
        assert len(new_log_lines) > 0, "No new logs were written"
        
        # Parse each new log line as JSON
        for line in new_log_lines:
            line = line.strip()
            if line:  # Skip empty lines
                # Strict JSON parsing (will fail on NaN, Infinity, etc.)
                try:
                    log_entry = json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(
                        f"Log line is not valid JSON: {line}\nError: {e}"
                    )
                
                # Verify expected fields
                assert "timestamp" in log_entry, "Missing 'timestamp' field"
                assert "client_id" in log_entry, "Missing 'client_id' field"
                assert "score" in log_entry, "Missing 'score' field"
                assert "prediction" in log_entry, "Missing 'prediction' field"
                assert "threshold" in log_entry, "Missing 'threshold' field"
                assert "inference_time_ms" in log_entry, "Missing 'inference_time_ms' field"
                assert "features" in log_entry, "Missing 'features' field"
                
                # Verify features dict doesn't contain NaN/Infinity
                features = log_entry["features"]
                assert isinstance(features, dict), "Features should be a dict"
                for feature_name, feature_value in features.items():
                    # None (null in JSON) is acceptable, but not NaN/Infinity
                    if feature_value is not None:
                        assert not (isinstance(feature_value, float) and 
                                  math.isnan(feature_value)), \
                            f"Feature {feature_name} contains NaN in logs"
                        assert not (isinstance(feature_value, float) and 
                                  math.isinf(feature_value)), \
                            f"Feature {feature_name} contains Infinity in logs"

"""Model service for credit scoring predictions."""

import logging
import random
import time
from typing import Any


logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and running the credit scoring model.
    
    Currently uses a mock model that returns random scores.
    Will be replaced with an actual ML model in production.
    """
    
    def __init__(self) -> None:
        """Initialize the model service.
        
        Loads the model from disk (mock implementation).
        """
        logger.info("Initializing ModelService")
        # Mock model initialization
        self.model = self._load_model()
        logger.info("Model loaded successfully")
    
    def _load_model(self) -> dict[str, Any]:
        """Load the model from disk.
        
        Returns:
            dict: Mock model object (placeholder for real model)
        """
        # Placeholder for actual model loading
        # In production, this would load from model/model.pkl or model/model.joblib
        logger.info("Loading mock model")
        return {"type": "mock", "version": "0.1.0"}
    
    def predict(self, features: dict[str, Any]) -> float:
        """Generate a prediction for the given features.
        
        Args:
            features: Dictionary of input features for the model
            
        Returns:
            float: Default probability score between 0 and 1
        """
        start_time = time.time()
        
        # Mock prediction: random score
        score = random.uniform(0, 1)
        
        inference_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Prediction completed",
            extra={
                "inference_time_ms": inference_time_ms,
                "score": score
            }
        )
        
        return score


# Singleton instance
model_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Get or create the singleton model service instance.
    
    Returns:
        ModelService: The global model service instance
    """
    global model_service
    if model_service is None:
        model_service = ModelService()
    return model_service


def initialize_model_service() -> ModelService:
    """Initialize the model service (for startup events).
    
    Returns:
        ModelService: The initialized model service instance
    """
    global model_service
    model_service = ModelService()
    return model_service

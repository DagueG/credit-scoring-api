"""Model service for credit scoring predictions with real ML model."""

import logging
import os
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and running the credit scoring model.
    
    Loads a pre-trained sklearn Pipeline with LightGBM and reference data
    for feature lookup.
    """
    
    def __init__(
        self,
        model_path: str = "model/model.pkl",
        reference_path: str = "data/clients_reference.parquet"
    ) -> None:
        """Initialize the model service.
        
        Loads the model and reference data into memory.
        
        Args:
            model_path: Path to the pickled sklearn Pipeline model
            reference_path: Path to the reference dataset (parquet format)
            
        Raises:
            FileNotFoundError: If model or reference data files don't exist
        """
        logger.info("Initializing ModelService")
        
        # Load the model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        logger.info("✓ Model loaded successfully")
        
        # Load reference data
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference data not found: {reference_path}")
        
        logger.info(f"Loading reference data from {reference_path}")
        self.clients_df = pd.read_parquet(reference_path)
        logger.info(f"✓ Reference data loaded: {len(self.clients_df)} clients")
        
        # Get expected features from the model
        if hasattr(self.model, 'feature_names_in_'):
            self.expected_features = list(self.model.feature_names_in_)
            logger.info(
                f"Model expects {len(self.expected_features)} features"
            )
        else:
            raise ValueError("Model does not have 'feature_names_in_' attribute")
    
    def get_client_features(self, client_id: int) -> pd.DataFrame:
        """Get features for a specific client.
        
        Args:
            client_id: The SK_ID_CURR client identifier
            
        Returns:
            pd.DataFrame: Single row DataFrame with client features
            
        Raises:
            KeyError: If client_id not found in reference data
        """
        if client_id not in self.clients_df.index:
            raise KeyError(f"Client {client_id} not found in reference data")
        
        # Return as DataFrame (required by sklearn Pipeline)
        return self.clients_df.loc[[client_id]]
    
    def predict(self, client_id: int) -> dict[str, Any]:
        """Generate a prediction for the given client.
        
        Args:
            client_id: The SK_ID_CURR client identifier
            
        Returns:
            dict: Contains keys:
                - score: float (probability of default, 0-1)
                - prediction: int (0 or 1)
                - inference_time_ms: float (time taken for inference)
                - features: dict (features used for prediction)
                
        Raises:
            KeyError: If client not found
            ValueError: If model inference fails
        """
        start_time = time.time()
        
        try:
            # Get client features
            X = self.get_client_features(client_id)
            features_dict = X.to_dict(orient='records')[0]
            # Convert numpy/pandas types to Python native types for JSON serialization
            features_dict = {
                k: (int(v) if isinstance(v, (int, np.integer)) else 
                    float(v) if isinstance(v, (float, np.floating)) else v)
                for k, v in features_dict.items()
            }
            
            # Run inference
            # predict_proba returns [[P(0), P(1)], ...]
            # We want P(1) which is the default probability
            proba = self.model.predict_proba(X)
            score = float(proba[0][1])  # Probability of class 1 (default)
            
            # Get prediction (0 or 1)
            prediction = self.model.predict(X)[0]
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"Prediction for client {client_id}: "
                f"score={score:.4f}, prediction={prediction}, "
                f"time={inference_time_ms:.2f}ms"
            )
            
            return {
                "score": score,
                "prediction": int(prediction),
                "inference_time_ms": inference_time_ms,
                "features": features_dict
            }
            
        except KeyError as e:
            logger.warning(f"Client not found: {client_id}")
            raise
        except Exception as e:
            logger.error(f"Prediction failed for client {client_id}: {str(e)}", exc_info=True)
            raise ValueError(f"Prediction failed: {str(e)}")


# Singleton instance
model_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Get or create the singleton model service instance.
    
    Returns:
        ModelService: The global model service instance
        
    Raises:
        RuntimeError: If model service not initialized
    """
    global model_service
    if model_service is None:
        raise RuntimeError(
            "Model service not initialized. Call initialize_model_service() at startup."
        )
    return model_service


def initialize_model_service(
    model_path: str = "model/model.pkl",
    reference_path: str = "data/clients_reference.parquet"
) -> ModelService:
    """Initialize the model service (for startup events).
    
    Args:
        model_path: Path to the pickled model
        reference_path: Path to the reference dataset
        
    Returns:
        ModelService: The initialized model service instance
    """
    global model_service
    model_service = ModelService(model_path=model_path, reference_path=reference_path)
    return model_service

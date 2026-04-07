"""FastAPI application for credit scoring predictions."""

import json
import logging
import os
import time
import random
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.model import initialize_model_service, get_model_service
from app.schemas import CreditRequest, CreditResponse


# Custom JSON encoder for numpy types and NaN handling
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and NaN/Infinity values.
    
    Converts:
    - numpy integer types → Python int
    - numpy float types → Python float
    - numpy arrays → list
    - NaN values (numpy.nan, float('nan'), pd.NA) → None (becomes null in JSON)
    - Infinity values (numpy.inf, float('inf')) → None (invalid in JSON)
    - NaT values → None
    """
    def default(self, obj):
        # Handle NaN/NaT values
        if pd.isna(obj):
            return None
        if isinstance(obj, np.nan.__class__):
            return None
        # Handle numpy integer types
        if isinstance(obj, np.integer):
            return int(obj)
        # Handle numpy float types
        if isinstance(obj, np.floating):
            val = float(obj)
            # Ensure no NaN or Infinity slips through
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        # Handle numpy arrays - recursively process to handle NaN/Inf in arrays
        if isinstance(obj, np.ndarray):
            result = obj.tolist()
            # If array contains NaN/Inf values, convert them to None
            if isinstance(result, list):
                result = [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x 
                         for x in result]
            return result
        return super().default(obj)


def clean_for_json(obj):
    """Recursively clean Python objects to remove NaN/Infinity values.
    
    Args:
        obj: Any Python object (dict, list, scalar, etc.)
        
    Returns:
        Cleaned object with NaN/Infinity replaced with None
    """
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        return clean_for_json(float(obj) if isinstance(obj, np.floating) else int(obj))
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    else:
        return obj


# Configure logging with JSON format to file
log_formatter = logging.Formatter(
    fmt='%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# File handler for API logs (JSON format)
file_handler = logging.FileHandler('logs/api_logs.jsonl')
file_handler.setFormatter(log_formatter)

# Console handler for general logging
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

# Get app logger
logger = logging.getLogger(__name__)


# Configuration
SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.5"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle.
    
    Startup: Initialize the model service
    Shutdown: Cleanup
    """
    # Startup
    logger.info(f"Starting Credit Scoring API with threshold={SCORE_THRESHOLD}")
    try:
        initialize_model_service()
        logger.info("✓ Model service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model service: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Credit Scoring API")


app = FastAPI(
    title="Credit Scoring API",
    description="API for credit scoring and risk assessment using real ML model",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint.
    
    Returns:
        dict: Status of the API
    """
    return {"status": "healthy"}


@app.get("/clients/sample", tags=["Clients"])
async def get_sample_clients(count: int = 10) -> dict[str, list[int]]:
    """Get a sample of valid client IDs for testing.
    
    Args:
        count: Number of sample client IDs to return (default 10)
        
    Returns:
        dict: List of valid client IDs
    """
    try:
        model_service = get_model_service()
        available_ids = model_service.clients_df.index.tolist()
        sample_ids = random.sample(available_ids, min(count, len(available_ids)))
        return {"client_ids": sorted(sample_ids)}
    except Exception as e:
        logger.error(f"Failed to get sample clients: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get sample clients")


@app.post("/predict", tags=["Predictions"], response_model=CreditResponse)
async def predict(request: CreditRequest) -> CreditResponse:
    """Predict credit scoring probability for a client.
    
    Args:
        request: CreditRequest containing client_id
        
    Returns:
        CreditResponse: Prediction result with score and probability
        
    Raises:
        HTTPException: If prediction fails or client not found
    """
    client_id = request.client_id
    start_time = time.time()
    
    try:
        model_service = get_model_service()
        
        # Get prediction from model
        prediction_result = model_service.predict(client_id)
        score = prediction_result["score"]
        prediction = prediction_result["prediction"]
        inference_time_ms = prediction_result["inference_time_ms"]
        features = prediction_result["features"]
        
        # Determine final prediction based on threshold
        final_prediction = 1 if score >= SCORE_THRESHOLD else 0
        
        # Create response
        response = CreditResponse(
            client_id=client_id,
            score=score,
            prediction=final_prediction,
            threshold=SCORE_THRESHOLD,
            inference_time_ms=inference_time_ms
        )
        
        # Log to JSON file
        log_entry = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "client_id": client_id,
            "score": score,
            "prediction": final_prediction,
            "threshold": SCORE_THRESHOLD,
            "inference_time_ms": inference_time_ms,
            "features": features
        }
        # Clean data to remove NaN/Infinity before JSON serialization
        clean_log_entry = clean_for_json(log_entry)
        file_handler.handle(
            logging.LogRecord(
                name=__name__,
                level=logging.INFO,
                pathname=__file__,
                lineno=0,
                msg=json.dumps(clean_log_entry, cls=NumpyEncoder, allow_nan=False),
                args=(),
                exc_info=None
            )
        )
        
        logger.info(
            f"Prediction successful - client_id={client_id}, score={score:.4f}, "
            f"prediction={final_prediction}, inference_time_ms={inference_time_ms:.2f}"
        )
        
        return response
        
    except KeyError:
        logger.warning(f"Client not found: {client_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Client with ID {client_id} not found in reference data"
        )
    except ValidationError as e:
        logger.error(f"Validation error in request: {e}")
        raise HTTPException(status_code=422, detail="Invalid request data")
    except Exception as e:
        logger.error(
            f"Prediction failed for client {client_id}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Prediction failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

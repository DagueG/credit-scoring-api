"""FastAPI application for credit scoring predictions."""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from app.model import initialize_model_service
from app.schemas import CreditRequest, CreditResponse


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
    initialize_model_service()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Credit Scoring API")


app = FastAPI(
    title="Credit Scoring API",
    description="API for credit scoring and risk assessment",
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


@app.post("/predict", tags=["Predictions"], response_model=CreditResponse)
async def predict(request: CreditRequest) -> CreditResponse:
    """Predict credit scoring probability.
    
    Args:
        request: CreditRequest containing input features
        
    Returns:
        CreditResponse: Prediction result with score and probability
        
    Raises:
        HTTPException: If prediction fails
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Import model service here to get the initialized instance
        from app.model import get_model_service
        model_service = get_model_service()
        
        # Convert request to dict for model input
        features = request.model_dump()
        
        # Get prediction
        score = model_service.predict(features)
        
        # Determine prediction based on threshold
        prediction = 1 if score >= SCORE_THRESHOLD else 0
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Generate response
        response = CreditResponse(
            client_id=request_id,
            score=score,
            prediction=prediction,
            threshold=SCORE_THRESHOLD
        )
        
        # Log to JSON file
        log_entry = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "request_id": request_id,
            "input_features": features,
            "score": score,
            "prediction": prediction,
            "threshold": SCORE_THRESHOLD,
            "inference_time_ms": inference_time_ms
        }
        file_handler.handle(
            logging.LogRecord(
                name=__name__,
                level=logging.INFO,
                pathname=__file__,
                lineno=0,
                msg=json.dumps(log_entry),
                args=(),
                exc_info=None
            )
        )
        
        logger.info(
            f"Prediction successful - request_id={request_id}, score={score:.4f}, "
            f"prediction={prediction}, inference_time_ms={inference_time_ms:.2f}"
        )
        
        return response
        
    except ValidationError as e:
        logger.error(f"Validation error in request: {e}")
        raise HTTPException(status_code=422, detail="Invalid request data")
    except Exception as e:
        logger.error(f"Prediction failed for request {request_id}: {str(e)}", exc_info=True)
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

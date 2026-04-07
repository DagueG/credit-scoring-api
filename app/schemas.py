"""Pydantic schemas for credit scoring API."""

from pydantic import BaseModel, Field


class CreditRequest(BaseModel):
    """Request schema for credit scoring prediction.
    
    Contains the client ID to look up features and generate a prediction.
    """
    
    client_id: int = Field(
        ...,
        description="Client ID (SK_ID_CURR from the database)",
        example=100001
    )


class CreditResponse(BaseModel):
    """Response schema for credit scoring prediction.
    
    Contains the scoring result and prediction details.
    """
    
    client_id: int = Field(
        ...,
        description="Client ID used for the prediction",
        example=100001
    )
    score: float = Field(
        ...,
        description="Default probability score (0 to 1) - P(class=1)",
        example=0.342
    )
    prediction: int = Field(
        ...,
        description="Final prediction (0=approved, 1=denied)",
        example=0
    )
    threshold: float = Field(
        ...,
        description="Score threshold used for prediction",
        example=0.5
    )
    inference_time_ms: float = Field(
        ...,
        description="Model inference time in milliseconds",
        example=12.5
    )

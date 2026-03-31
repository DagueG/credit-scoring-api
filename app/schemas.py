"""Pydantic schemas for credit scoring API."""

from pydantic import BaseModel, Field


class CreditRequest(BaseModel):
    """Request schema for credit scoring prediction.
    
    Contains the features required for credit risk assessment.
    """
    
    EXT_SOURCE_1: float = Field(
        ...,
        description="External data source 1",
        example=0.5019415614426349
    )
    EXT_SOURCE_2: float = Field(
        ...,
        description="External data source 2",
        example=0.3662832999999999
    )
    EXT_SOURCE_3: float = Field(
        ...,
        description="External data source 3",
        example=0.7119315
    )
    AMT_CREDIT: float = Field(
        ...,
        description="Credit amount in currency",
        example=179055.0
    )
    AMT_INCOME_TOTAL: float = Field(
        ...,
        description="Total income",
        example=112500.0
    )
    AMT_ANNUITY: float = Field(
        ...,
        description="Annuity amount",
        example=3951.0
    )
    DAYS_BIRTH: int = Field(
        ...,
        description="Client age in days (negative value)",
        example=-13363
    )
    DAYS_EMPLOYED: int = Field(
        ...,
        description="Number of days employed (negative value)",
        example=-640
    )


class CreditResponse(BaseModel):
    """Response schema for credit scoring prediction.
    
    Contains the scoring result and prediction.
    """
    
    client_id: str = Field(
        ...,
        description="Unique client identifier",
        example="CLIENT_001"
    )
    score: float = Field(
        ...,
        description="Default probability score (0 to 1)",
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

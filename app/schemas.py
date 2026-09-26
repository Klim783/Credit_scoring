# app/schemas.py
from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    NAME_CONTRACT_TYPE: str = Field(..., json_schema_extra={"example": "Cash loans"})
    CODE_GENDER: str = Field(..., json_schema_extra={"example": "M"})
    FLAG_OWN_CAR: str = Field(..., json_schema_extra={"example": "Y"})
    FLAG_OWN_REALTY: str = Field(..., json_schema_extra={"example": "Y"})
    CNT_CHILDREN: int = Field(..., json_schema_extra={"example": 0})
    AMT_INCOME_TOTAL: float = Field(..., json_schema_extra={"example": 150000.0})
    AMT_CREDIT: float = Field(..., json_schema_extra={"example": 500000.0})
    AMT_ANNUITY: float = Field(..., json_schema_extra={"example": 25000.0})
    AMT_GOODS_PRICE: float = Field(..., json_schema_extra={"example": 450000.0})
    NAME_TYPE_SUITE: str = Field(..., json_schema_extra={"example": "Unaccompanied"})
    NAME_INCOME_TYPE: str = Field(..., json_schema_extra={"example": "Working"})
    NAME_EDUCATION_TYPE: str = Field(..., json_schema_extra={"example": "Secondary / secondary special"})
    NAME_FAMILY_STATUS: str = Field(..., json_schema_extra={"example": "Married"})
    NAME_HOUSING_TYPE: str = Field(..., json_schema_extra={"example": "House / apartment"})
    DAYS_BIRTH: int = Field(..., json_schema_extra={"example": -12000})
    DAYS_EMPLOYED: int = Field(..., json_schema_extra={"example": -2000})
    DAYS_REGISTRATION: int = Field(..., json_schema_extra={"example": -4000})
    DAYS_ID_PUBLISH: int = Field(..., json_schema_extra={"example": -1500})
    EXT_SOURCE_1: float = Field(..., json_schema_extra={"example": 0.5})
    EXT_SOURCE_2: float = Field(..., json_schema_extra={"example": 0.6})
    EXT_SOURCE_3: float = Field(..., json_schema_extra={"example": 0.7})


class PredictionResponse(BaseModel):
    default_probability: float
    decision: str
    threshold: float
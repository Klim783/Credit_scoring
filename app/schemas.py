from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    NAME_CONTRACT_TYPE: str = Field(..., example="Cash loans")
    CODE_GENDER: str = Field(..., example="M")
    FLAG_OWN_CAR: str = Field(..., example="Y")
    FLAG_OWN_REALTY: str = Field(..., example="Y")
    CNT_CHILDREN: int = Field(..., example=0)
    AMT_INCOME_TOTAL: float = Field(..., example=150000.0)
    AMT_CREDIT: float = Field(..., example=500000.0)
    AMT_ANNUITY: float = Field(..., example=25000.0)
    AMT_GOODS_PRICE: float = Field(..., example=450000.0)
    NAME_TYPE_SUITE: str = Field(..., example="Unaccompanied")
    NAME_INCOME_TYPE: str = Field(..., example="Working")
    NAME_EDUCATION_TYPE: str = Field(
        ..., example="Secondary / secondary special"
    )
    NAME_FAMILY_STATUS: str = Field(..., example="Married")
    NAME_HOUSING_TYPE: str = Field(..., example="House / apartment")
    DAYS_BIRTH: int = Field(..., example=-12000)
    DAYS_EMPLOYED: int = Field(..., example=-2000)
    DAYS_REGISTRATION: int = Field(..., example=-4000)
    DAYS_ID_PUBLISH: int = Field(..., example=-1500)
    EXT_SOURCE_1: float = Field(..., example=0.5)
    EXT_SOURCE_2: float = Field(..., example=0.6)
    EXT_SOURCE_3: float = Field(..., example=0.7)

class PredictionResponse(BaseModel):
	default_probability: float
	decision: str
	threshold: float
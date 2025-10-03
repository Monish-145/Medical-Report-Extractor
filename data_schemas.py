from pydantic import BaseModel, Field, validator
from typing import List, Optional
import re

class TestResult(BaseModel):
    name: str = Field(..., description="Name of the laboratory test")
    value: str = Field(..., description="Numeric or textual result")
    unit: Optional[str] = Field(default="", description="Measurement unit")
    matched_tokens: List[str] = []
    confidence: float = Field(default=1.0, description="Confidence score of extraction (0–1)")

    @validator("unit", pre=True, always=True)
    def normalize_unit(cls, v):
        """Normalize common OCR mistakes in lab units"""
        if not v:
            return ""
        v = v.strip().lower()
        v = v.replace("dI", "dl").replace("ldl", "dl")
        v = v.replace("mgldl", "mg/dl")
        return v

    @validator("confidence", pre=True, always=True)
    def validate_confidence(cls, v):
        try:
            val = float(v)
            return min(max(val, 0.0), 1.0)  # clamp between 0 and 1
        except:
            return 0.0


class Patient(BaseModel):
    name: str = Field(default="UNKNOWN", description="Full patient name")
    age: Optional[int] = Field(default=None, description="Age in years")
    sex: str = Field(default="UNKNOWN", description="M/F/UNKNOWN")
    id: Optional[str] = Field(default=None, description="Hospital MRD or patient ID")
    dob: str = Field(default="UNKNOWN", description="Date of birth in dd/mm/yyyy format")
    visit_id: str = Field(default="UNKNOWN", description="Hospital visit identifier")
    date: str = Field(default="UNKNOWN", description="Report date in dd/mm/yyyy format")

    @validator("sex", pre=True, always=True)
    def normalize_sex(cls, v):
        if not v:
            return "UNKNOWN"
        v = str(v).strip().upper()[0]
        return v if v in ["M", "F"] else "UNKNOWN"

    @validator("dob", "date", pre=True)
    def validate_dates(cls, v):
        if not v or v == "UNKNOWN":
            return "UNKNOWN"
        v = str(v).strip()
        if re.match(r"\d{2}/\d{2}/\d{4}", v):
            return v
        return "UNKNOWN"

    @validator("age", pre=True)
    def validate_age(cls, v):
        try:
            return int(v)
        except:
            return None


class Report(BaseModel):
    patient: Patient
    tests: List[TestResult]

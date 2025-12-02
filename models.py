from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import Column, Integer, String
from database import Base

class UserCreate(BaseModel):
    name: str
    id:str
    email: EmailStr
    cellphone_number: str 
    password: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    cellphone_number: Optional[str]

    class Config:
        orm_mode = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str



class MetricsCreate(BaseModel):
    heart_rate: float
    activity_level: str
    stress_level: float
    timestamp: datetime


class MetricsResponse(BaseModel):
    id: int
    heart_rate: float
    activity_level: str
    stress_level: float
    timestamp: datetime

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    id: Optional[int] = None


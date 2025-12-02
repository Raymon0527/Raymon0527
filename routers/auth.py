from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.security import OAuth2PasswordBearer
from database import get_db
from models_db import User, create_metrics_table_for_user
from models import UserCreate, UserLogin, Token, UserResponse
from jose import jwt
from datetime import timedelta
from fastapi import Body
from pydantic import BaseModel, EmailStr
from utils.auth_utils import hash_password, verify_password, create_access_token,get_current_user 

router = APIRouter()


class SignupRequest(BaseModel):
    full_name: str
    username: str
    student_id: str
    email: EmailStr
    password: str
    confirm_password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/signup", response_model=Token)
def signup(user: SignupRequest, db: Session = Depends(get_db)):
    # 1. Check if passwords match
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    # 2. Check if email already exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 3. Check if username already exists
    existing_username = db.query(User).filter(User.username == user.username).first()
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already taken")

    # 4. Hash password
    hashed_pw = hash_password(user.password)

    # 5. Create new user
    new_user = User(
        full_name=user.full_name,
        username=user.username,
        student_id=user.student_id,
        email=user.email,
        password=hashed_pw,
        phone=None,
        emergency_contact=None,
        avatar_url=None 
 )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # 6. Generate token
    access_token = create_access_token(data={"sub": str(new_user.id)})

    return {"access_token": access_token, "token_type": "bearer"}




# --- LOGIN ROUTE 
@router.post("/login", response_model=Token)
def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_credentials.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not verify_password(user_credentials.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me")
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """
    Returns the currently logged-in user's profile info, including optional fields.
    """
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "username": current_user.username,
        "student_id": current_user.student_id,
        "email": current_user.email,
        "phone": current_user.phone,                      # optional
        "emergency_contact": current_user.emergency_contact,  # optional
        "avatar_url": current_user.avatar_url             # optional
    }

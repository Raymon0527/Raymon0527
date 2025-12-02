from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from database import get_db
from models_db import User, create_metrics_table_for_user

# -------------------------------
# CONFIGURATION
# -------------------------------

SECRET_KEY = "my_secret_key"  # <- replace with your own generated key
ALGORITHM = "HS256" 
ACCESS_TOKEN_EXPIRE_MINUTES = 600

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# PASSWORD HELPERS

def hash_password(password: str) -> str:
    """Hash a plain password securely using bcrypt."""
    if len(password.encode("utf-8")) > 72:
        password = password[:72]  # bcrypt only considers the first 72 bytes
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against its hashed version."""
    return pwd_context.verify(plain_password, hashed_password)

# TOKEN CREATION

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token with expiration and a user identifier (sub)."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})

    # Ensure 'sub' field exists
    if "sub" not in to_encode and "id" in to_encode:
        to_encode["sub"] = str(to_encode["id"])

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# GET CURRENT USER

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        print("🔍 Token received:", token)
        #  Remove 'Bearer ' prefix if it exists
        if token.startswith("Bearer "):
            token = token.split(" ")[1]

        payload = jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)
        print("Payload decoded:", payload)

        user_id = payload.get("sub")
        if user_id is None:
            print("'sub' field missing in payload")
            raise credentials_exception

    except JWTError as e:
        print("JWTError:", e)
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        print("User not found in DB for id:", user_id)
        raise credentials_exception

    print("Authenticated user:", user.email)
    return user


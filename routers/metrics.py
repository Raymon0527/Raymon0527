from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from sqlalchemy import func, create_engine, insert, select, text
from database import get_db, Base, engine
from models_db import create_metrics_table_for_user, User
from utils.auth_utils import get_current_user
from typing import List
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer

router = APIRouter()
engine = create_engine("sqlite:///student.db", echo=False)



router = APIRouter(prefix="/metrics", tags=["Metrics"])
@router.get("/latest")
def get_latest_metrics(current_user: dict = Depends(get_current_user)):
    user_id = current_user.id  
    table_name = f"metrics_user{user_id}"

    if not engine.dialect.has_table(engine.connect(), table_name):
        raise HTTPException(status_code=404, detail="No metrics found for this user")

    try:
        with engine.connect() as conn:
            query = text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 3")
            results = conn.execute(query).fetchall()

        if not results:
            raise HTTPException(status_code=404, detail="No data available")

        return [dict(row._mapping) for row in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




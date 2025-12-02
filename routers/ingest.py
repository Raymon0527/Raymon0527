from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import insert
from pydantic import BaseModel
from datetime import datetime
from database import get_db
from models_db import create_metrics_table_for_user
from utils.auth_utils import get_current_user

router = APIRouter(prefix="/ingest", tags=["Ingest"])

# Data model for request body
class MetricsData(BaseModel):
    heart_rate: float
    motion_intensity: float
    

@router.post("/")
def ingest_sensor(
    data: MetricsData,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """
    Saves data into the logged-in user's own metrics table.
    """

    user_id = current_user.id  

    try:
        user_metrics = create_metrics_table_for_user(user_id=user_id, db=db)

        
        db.execute(
            insert(user_metrics).values(
                heart_rate=data.heart_rate,
                motion_intensity=data.motion_intensity,
                timestamp=datetime.utcnow(),
            )
        )

        
        db.commit()

        return {"message": f"Metrics data saved successfully for user {user_id}."}

    except Exception as e:
        db.rollback()
        print("Error during ingestion:", e)
        raise HTTPException(status_code=500, detail=str(e))

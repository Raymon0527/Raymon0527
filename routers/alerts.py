from fastapi import APIRouter
from calculation import generate_alerts

router = APIRouter(prefix="/alerts", tags=["Alerts"])

@router.get("/")
def get_alerts():
    return {"alerts": generate_alerts()}

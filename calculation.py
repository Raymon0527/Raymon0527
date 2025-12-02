from sqlalchemy.orm import Session
from models_db import User, create_metrics_table_for_user


def latest_metrics(db: Session):
    """Fetch the latest sensor metrics from the database"""
    latest = db.query(User).order_by(User.timestamp.desc()).first()
    if not latest:
        return {"heart_rate": None, "activity_level": None, "stress_level": None}

    return {
        "heart_rate": latest.heart_rate,
        "activity_level": latest.activity_level,
        "stress_level": latest.stress_level,
        "timestamp": latest.timestamp
    }


def generate_alerts(db: Session):
    """Generate alerts if abnormal heart rate or stress levels are detected"""
    alerts = []
    students = db.query(User).all()

    for data in students:
        if data.heart_rate > 100:
            alerts.append(f"High heart rate detected for {data.student_name}")
        if data.stress_level > 80:
            alerts.append(f"High stress level detected for {data.student_name}")

    return alerts

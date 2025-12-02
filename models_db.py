
from sqlalchemy import Column, Integer, String, Float, DateTime, Table, MetaData, inspect
from sqlalchemy.orm import Session
from datetime import datetime
from database import Base, engine

metadata = MetaData()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    student_id = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    emergency_contact = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)


def create_metrics_table_for_user(user_id: int, db: Session):
    """
    Dynamically creates a new metrics table for each user.
    Example: metrics_user12
    """

    table_name = f"metrics_user{user_id}"

    inspector = inspect(db.bind)
    if inspector.has_table(table_name):
        print(f"Table '{table_name}' already exists.")
        return Table(table_name, metadata, autoload_with=db.bind)

    user_metrics = Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, index=True),
        Column("heart_rate", Float, nullable=False),
        Column("motion", String, nullable=False),
        Column("ir_value", Float, nullable=False),
        Column("timestamp", DateTime, default=datetime.utcnow),
    )

    metadata.create_all(db.bind, tables=[user_metrics])
    print(f" Created table: {table_name}")

    return user_metrics











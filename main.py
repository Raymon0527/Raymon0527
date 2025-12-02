from fastapi import FastAPI
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.security import OAuth2PasswordBearer
from routers import ingest, metrics, alerts, auth, ai_model
from database import Base, engine
from models_db import User, create_metrics_table_for_user
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from routers.auth import router as auth_router


Base.metadata.create_all(bind=engine)
app = FastAPI (title="IoT Health & Activity API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Health Metrics API",
        version="1.0.0",
        description="API for user authentication and heart metrics tracking",
        routes=app.routes,
    )
    
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Register routers
app.include_router(ingest.router)
app.include_router(metrics.router) 
app.include_router(alerts.router) 
app.include_router(auth.router, prefix="/auth")
app.include_router(ai_model.router)

@app.get("/")
def read_root():
    return {"message": "IoT Backend is running"}






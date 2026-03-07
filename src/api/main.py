# FastAPI entry point
from fastapi import FastAPI
app = FastAPI(title="AI Ocean Platform API")

@app.get("/health")
def health():
    return {"status": "ok"}

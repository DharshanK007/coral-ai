# FastAPI entry point
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AI Ocean Platform API")

# Configure CORS for local development (Vite runs on port 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/data")
def get_dashboard_data():
    """
    Serve the AI-generated JSON data over a REST endpoint.
    This replaces the old method of writing directly to a Javascript file.
    """
    # The pipeline script writes to data/processed/dashboard_data.json
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed", "dashboard_data.json")
    
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(
            content=data,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )
    else:
        # Fallback if pipeline hasn't run yet
        return JSONResponse(content={"error": "Data not found. Please run the pipeline exporter first."}, status_code=404)

# Mount the static frontend directory so it runs on the same port as the API
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
frontend_path = os.path.join(root_dir, "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


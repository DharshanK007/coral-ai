# FastAPI entry point
import json
import os
import mimetypes
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
import hashlib
import subprocess
import sys
import re
from pydantic import BaseModel

# Register MIME types explicitly to fix Windows registry and Docker slim missing /etc/mime.types
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".jsx")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/svg+xml", ".svg")

# --- Database Setup ---
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "users.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT)")
    try:
        conn.execute("ALTER TABLE users ADD COLUMN last_login TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

# Run schema initialization once when the app starts
init_db()

def get_db():
    return sqlite3.connect(DB_PATH, timeout=15)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

class User(BaseModel):
    username: str
    password: str
    email: str | None = None

class PipelineRun(BaseModel):
    start_date: str
    end_date: str

from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI(title="AI Ocean Platform API")

# Enable automatic Gzip compression for files larger than 1KB (reduces JS/CSS transfer size by ~70%)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configure CORS for local development (Vite runs on port 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware to disable caching on all GET requests (prevent old font/css/js caching)
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.method == "GET":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/register")
def register(user: User):
    if len(user.password) < 5 or not re.search(r"[!@#$%^&*]", user.password):
        raise HTTPException(status_code=400, detail="Password must be at least 5 characters long and include at least one special character.")
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", (user.username, hash_password(user.password), user.email))
        conn.commit()
        return {"success": True, "message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        conn.close()

@app.post("/api/login")
def login(user: User):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (user.username,))
    row = cursor.fetchone()
    if row and row[0] == hash_password(user.password):
        from datetime import datetime
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        cursor.execute("UPDATE users SET last_login = ? WHERE username = ?", (now_str, user.username))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Login successful", "username": user.username}
    conn.close()
    raise HTTPException(status_code=401, detail="Invalid credentials")

# Pipeline status — updated by polling the progress file pipeline.py writes
PIPELINE_STATUS = {
    "status": "idle",   # "idle", "running", "completed", "failed"
    "message": "",
    "error": "",
    "progress": 0,
    "stage": "",
    "completed_stages": []
}

# Path to the real-time progress file written by pipeline.py
_PROGRESS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "processed", "pipeline_progress.json"
)

def _poll_progress_file():
    """Read the latest stage from the progress file pipeline.py writes."""
    global PIPELINE_STATUS
    try:
        if os.path.exists(_PROGRESS_FILE):
            with open(_PROGRESS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            PIPELINE_STATUS["stage"]            = data.get("stage", "")
            PIPELINE_STATUS["progress"]         = data.get("progress", 0)
            PIPELINE_STATUS["message"]          = data.get("message", "")
            PIPELINE_STATUS["completed_stages"] = data.get("completed_stages", [])
    except Exception:
        pass  # never crash the poller

def run_pipeline_task(start_date: str, end_date: str):
    import time, threading
    global PIPELINE_STATUS

    # Clear any stale progress file from a previous run
    try:
        if os.path.exists(_PROGRESS_FILE):
            os.remove(_PROGRESS_FILE)
    except Exception:
        pass

    PIPELINE_STATUS["status"]           = "running"
    PIPELINE_STATUS["error"]            = ""
    PIPELINE_STATUS["progress"]         = 2
    PIPELINE_STATUS["stage"]            = "Starting Pipeline"
    PIPELINE_STATUS["message"]          = "Initialising neural pipeline environment..."
    PIPELINE_STATUS["completed_stages"] = []

    stop_polling = threading.Event()

    def poller():
        while not stop_polling.is_set():
            _poll_progress_file()
            time.sleep(1.5)

    poll_thread = threading.Thread(target=poller, daemon=True)
    poll_thread.start()

    try:
        pipeline_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline.py"
        )
        subprocess.run(
            [sys.executable, pipeline_script, "--start", start_date, "--end", end_date],
            check=True
        )
        # Do one final read before marking complete
        _poll_progress_file()
        PIPELINE_STATUS["status"]   = "completed"
        PIPELINE_STATUS["progress"] = 100
        PIPELINE_STATUS["stage"]    = "Complete"
        PIPELINE_STATUS["message"]  = "Pipeline completed successfully"
        # Mark every stage done
        from src.pipeline import _STAGE_ORDER
        PIPELINE_STATUS["completed_stages"] = list(_STAGE_ORDER)
    except Exception as e:
        PIPELINE_STATUS["status"]  = "failed"
        PIPELINE_STATUS["error"]   = str(e)
        PIPELINE_STATUS["message"] = "Pipeline execution failed"
    finally:
        stop_polling.set()

@app.post("/api/run_pipeline")
def api_run_pipeline(run: PipelineRun, background_tasks: BackgroundTasks):
    # --- HYBRID ARCHITECTURE SUPPORT ---
    if os.environ.get("DEPLOYMENT_MODE") == "cloud":
        raise HTTPException(
            status_code=503, 
            detail="Live Analysis is currently unavailable. The primary AI server (laptop) is offline. You are currently exploring the cached Cloud Dashboard."
        )
    # -----------------------------------
        
    global PIPELINE_STATUS
    if PIPELINE_STATUS["status"] == "running":
        raise HTTPException(status_code=400, detail="Pipeline is already running")
    
    background_tasks.add_task(run_pipeline_task, run.start_date, run.end_date)
    return {"success": True, "message": "Pipeline started in background", "status": "running"}

@app.get("/api/pipeline_status")
def get_pipeline_status():
    global PIPELINE_STATUS
    return PIPELINE_STATUS

@app.get("/api/data")
def get_dashboard_data():
    """
    Serve the AI-generated JSON data over a REST endpoint.
    This replaces the old method of writing directly to a Javascript file.
    """
    import math

    def sanitize_floats(obj):
        """Recursively replace NaN / Inf / -Inf with None so JSON stays valid."""
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: sanitize_floats(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_floats(v) for v in obj]
        return obj

    # The pipeline script writes to data/processed/dashboard_data.json
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed", "dashboard_data.json")
    
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = f.read()
        # json.loads tolerates NaN/Infinity written by Python's json.dumps(allow_nan=True)
        data = json.loads(raw)
        clean = sanitize_floats(data)
        return JSONResponse(
            content=clean,
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
frontend_dist_path = os.path.join(root_dir, "frontend", "dist")
frontend_src_path = os.path.join(root_dir, "frontend")

@app.get("/data.js")
def get_data_js():
    js_path = os.path.join(root_dir, "frontend", "data.js")
    if os.path.exists(js_path):
        from fastapi.responses import FileResponse
        return FileResponse(js_path, media_type="application/javascript")
    return JSONResponse(content={"error": "Not found"}, status_code=404)

@app.get("/")
def get_index_html():
    from fastapi.responses import FileResponse
    path = os.path.join(frontend_dist_path, "index.html") if os.path.exists(frontend_dist_path) else os.path.join(frontend_src_path, "index.html")
    if os.path.exists(path):
        return FileResponse(
            path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return JSONResponse(content={"error": "Not found"}, status_code=404)

@app.get("/globe.html")
def get_globe_html():
    from fastapi.responses import FileResponse
    path = os.path.join(frontend_dist_path, "globe.html") if os.path.exists(frontend_dist_path) else os.path.join(frontend_src_path, "globe.html")
    if os.path.exists(path):
        return FileResponse(
            path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return JSONResponse(content={"error": "Not found"}, status_code=404)

if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")
elif os.path.exists(frontend_src_path):
    app.mount("/", StaticFiles(directory=frontend_src_path, html=True), name="frontend")


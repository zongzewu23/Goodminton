from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tempfile

from .inference import run_inference_on_video

# FastAPI application
app = FastAPI()

# Frontend directory: src/frontend
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    # Serve the HTML landing page
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend not found: index.html")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/predict-clear")
async def predict_clear(file: UploadFile = File(...), model: str = Form("svm")):
    # Validate model choice
    if model not in ("svm", "lr_pca"):
        raise HTTPException(status_code=400, detail="Invalid model")

    # Persist the uploaded video temporarily for processing
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result = run_inference_on_video(tmp_path, model_type=model)
    except Exception as e:
        # Map any inference error to a 500 for now
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return JSONResponse(result)
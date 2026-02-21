import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import sys
import time

sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.tutor import HistoryTutor

class ChatRequest(BaseModel):
    message: str
    use_rag: Optional[bool] = None

class ModelSwitchRequest(BaseModel):
    model_type: str
    adapter_path: Optional[str] = None

tutor = None
current_model = "base"
rag_enabled = False
upload_dir = Path("./data/uploads")

@asynccontextmanager
async def lifespan(_):
    global tutor, upload_dir
    print("Initializing AI Tutor...")

    # Clean uploads directory on startup (except .gitkeep)
    if upload_dir.exists():
        for file in upload_dir.iterdir():
            if file.is_file() and file.name != '.gitkeep':
                file.unlink()
        print(f"Cleared {upload_dir}")
    upload_dir.mkdir(parents=True, exist_ok=True)

    tutor = HistoryTutor(config_path="./configs/config.yaml")
    tutor.setup_llm()
    print("Base model loaded and ready!")

    yield

    # Clean uploads on shutdown (except .gitkeep)
    print("Shutting down and cleaning up...")
    if upload_dir.exists():
        for file in upload_dir.iterdir():
            if file.is_file() and file.name != '.gitkeep':
                file.unlink()
        print(f"Cleaned up {upload_dir}")

app = FastAPI(title="AI Tutor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/api/chat/message")
async def chat(request: ChatRequest):
    try:
        use_rag = request.use_rag if request.use_rag is not None else rag_enabled
        response_text = tutor.chat(request.message, use_rag=use_rag)

        detected_intent = tutor.last_intent.value if tutor.last_intent else "unknown"
        detected_confidence = tutor.last_confidence

        # Use sources already retrieved during chat() — no second retrieval needed
        sources = [
            {"text": doc.text[:200], "score": score, "metadata": doc.metadata}
            for doc, score in tutor.last_sources
        ]

        return {
            "response": response_text,
            "intent": detected_intent,
            "confidence": detected_confidence,
            "sources": sources,
            "model_info": {"type": current_model, "rag_enabled": use_rag}
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/chat/history")
async def clear_history():
    tutor.reset()
    return {"success": True, "message": "History cleared"}

@app.get("/api/models/current")
async def get_current_model():
    return {
        "model_type": current_model,
        "rag_enabled": rag_enabled,
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "adapter_path": None,
        "memory_usage": None
    }

@app.post("/api/models/switch")
async def switch_model(request: ModelSwitchRequest):
    global current_model, rag_enabled
    try:
        start = time.time()
        model_type = request.model_type

        if model_type == "base":
            tutor.setup_llm()
            rag_enabled = False

        elif model_type == "base_rag":
            tutor.setup_llm()
            if not tutor.retriever:
                tutor.setup_rag(
                    dataset_path="./questions_dataset.json",
                    pdf_directory="./data",
                    index_name="history_index",
                    rebuild_index=False
                )
            rag_enabled = True

        elif model_type == "finetuned":
            adapter_path = request.adapter_path or "./models/finetuned_8b/final"
            tutor.setup_llm(adapter_path=adapter_path)
            if not tutor.retriever:
                tutor.setup_rag(
                    dataset_path="./questions_dataset.json",
                    pdf_directory="./data",
                    index_name="history_index",
                    rebuild_index=False
                )
            rag_enabled = True

        current_model = model_type
        load_time = time.time() - start

        return {
            "success": True,
            "model_type": model_type,
            "rag_enabled": rag_enabled,
            "message": f"Switched to {model_type}",
            "load_time_seconds": load_time
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        if not tutor.retriever:
            raise HTTPException(400, "Switch to base_rag or finetuned model first")

        uploaded_files = []
        for file in files:
            file_path = upload_dir / file.filename
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            if ".gitkeep" in file.filename:
                continue

            if file.filename.endswith('.pdf'):
                doc = tutor.doc_processor.load_pdf_file(str(file_path))
            else:
                doc = tutor.doc_processor.load_text_file(str(file_path))

            chunks = tutor.doc_processor.process_documents([doc], chunk=True)
            tutor.retriever.add_documents(chunks)

            uploaded_files.append({
                "filename": file.filename,
                "file_type": "pdf" if file.filename.endswith('.pdf') else "txt",
                "size_bytes": len(content),
                "chunks_created": len(chunks),
                "file_path": str(file_path)
            })

        tutor.retriever.save_index("history_index")

        return {
            "success": True,
            "uploaded_files": uploaded_files,
            "total_documents_in_index": tutor.retriever.index.ntotal,
            "message": f"Uploaded {len(files)} files"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/documents/list")
async def list_documents():
    documents = []
    if upload_dir.exists():
        for file_path in upload_dir.iterdir():
            if file_path.is_file() and file_path.name != '.gitkeep':
                documents.append({
                    "filename": file_path.name,
                    "file_type": "pdf" if file_path.suffix == '.pdf' else "txt",
                    "upload_date": "2024-01-01",
                    "size_bytes": file_path.stat().st_size,
                    "chunks": 0,
                    "source": "upload"
                })
    return {"documents": documents, "total_files": len(documents), "total_chunks": 0}

@app.post("/api/session/reset")
async def reset_session():
    tutor.reset()
    return {"success": True, "message": "Session reset"}

@app.get("/api/session/stats")
async def get_stats():
    return tutor.get_statistics()

@app.get("/api/session/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": tutor.generator is not None,
        "rag_initialized": tutor.retriever is not None,
        "uptime_seconds": 0
    }

@app.get("/api/health")
async def api_health():
    return {"status": "ok"}

web_dir = Path(__file__).parent.parent / "web"
app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

@app.get("/")
async def root():
    return FileResponse(str(web_dir / "templates" / "index.html"))

def main():
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    main()

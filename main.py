# main.py
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import httpx
from dotenv import load_dotenv

# load .env locally if present (won't override server env)
load_dotenv()

# ---------- Config / env ----------
MONGO_URI = os.getenv("MONGO_URI")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TEACHER_CHAT_ID = os.getenv("TEACHER_CHAT_ID")
TEACHER_API_KEY = os.getenv("TEACHER_API_KEY")  # optional simple auth
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional for future

if not MONGO_URI:
    logging.warning("MONGO_URI not set — DB features will fail until provided.")
if not TELEGRAM_BOT_TOKEN:
    logging.warning("TELEGRAM_BOT_TOKEN not set — telegram notifications will be disabled.")
if not TEACHER_CHAT_ID:
    logging.warning("TEACHER_CHAT_ID not set — teacher notifications may fail.")

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("app")

# ---------- FastAPI app ----------
app = FastAPI(title="Zynno - Doubt backend")

# CORS: adjust allowed origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Mongo connection ----------
mongo_client: Optional[AsyncIOMotorClient] = None
db = None

async def get_db():
    return db

# ---------- Pydantic models ----------
class DoubtIn(BaseModel):
    student_name: str = Field(..., min_length=1)
    student_class: str = Field(..., min_length=1)
    doubt_text: str = Field(..., min_length=1)

class DoubtOut(DoubtIn):
    id: str
    status: str
    created_at: datetime
    teacher_reply: Optional[str] = None
    notify_status: Optional[str] = None  # success/failed/queued

# ---------- Helpers ----------
def serialize_doc(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    _id = out.pop("_id", None)
    out["id"] = str(_id) if _id else None
    # convert datetimes to iso
    if isinstance(out.get("created_at"), datetime):
        out["created_at"] = out["created_at"].isoformat()
    return out

# ---------- Simple in-memory rate limiter (per-ip) ----------
RATE_LIMIT_MAX = 20  # requests
RATE_LIMIT_WINDOW = timedelta(minutes=60)
_rate_store: Dict[str, List[datetime]] = {}

def check_rate_limit(ip: str) -> bool:
    now = datetime.utcnow()
    arr = _rate_store.get(ip, [])
    # remove old
    arr = [t for t in arr if now - t <= RATE_LIMIT_WINDOW]
    if len(arr) >= RATE_LIMIT_MAX:
        _rate_store[ip] = arr
        return False
    arr.append(now)
    _rate_store[ip] = arr
    return True

# ---------- Telegram notifier with retries ----------
async def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        logger.warning("Telegram token/chat id missing")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(1, 4):
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    logger.info("Telegram sent")
                    return True
                else:
                    logger.warning("Telegram failed status=%s body=%s", resp.status_code, resp.text)
            except Exception as e:
                logger.exception("Telegram request error: %s", e)
            await asyncio.sleep(1 * attempt)
    return False

# ---------- Auth dependency for teacher endpoints ----------
def require_teacher_api_key(x_api_key: Optional[str] = Header(None)):
    if TEACHER_API_KEY:
        if not x_api_key or x_api_key != TEACHER_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    # if TEACHER_API_KEY not set, allow but log warning
    else:
        logger.warning("TEACHER_API_KEY not configured — teacher endpoints open")

# ---------- Startup / shutdown ----------
@app.on_event("startup")
async def startup():
    global mongo_client, db
    if MONGO_URI:
        mongo_client = AsyncIOMotorClient(MONGO_URI)
        # pick database name from URI or set default
        db_name = os.getenv("MONGO_DB_NAME") or "zynno_db"
        db = mongo_client[db_name]
        # ensure indexes
        try:
            await db.doubts.create_index([("status", 1)])
            await db.doubts.create_index([("created_at", -1)])
            await db.doubts.create_index([("student_class", 1)])
            logger.info("Indexes created")
        except Exception as e:
            logger.exception("Index creation failed: %s", e)
    else:
        logger.warning("No MONGO_URI — DB not connected")

@app.on_event("shutdown")
async def shutdown():
    global mongo_client
    if mongo_client:
        mongo_client.close()

# ---------- Health endpoint ----------
@app.get("/health")
async def health():
    ok = {"status": "ok", "db": False}
    try:
        if db:
            await db.command("ping")
            ok["db"] = True
    except Exception as e:
        logger.warning("Health check DB failed: %s", e)
    return ok

# ---------- Doubt endpoints ----------
@app.post("/doubts", response_model=Dict[str, Any])
async def create_doubt(request: Request, payload: DoubtIn, background_tasks: BackgroundTasks, db=Depends(get_db)):
    # rate-limit by client IP
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")

    doc = payload.dict()
    doc.update({
        "status": "open",
        "created_at": datetime.utcnow(),
        "teacher_reply": None,
        "notify_status": "queued"
    })

    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")

    res = await db.doubts.insert_one(doc)
    doubt_id = str(res.inserted_id)

    # prepare telegram text
    text = (
        f"<b>New Doubt</b>\n"
        f"Student: {payload.student_name}\n"
        f"Class: {payload.student_class}\n\n"
        f"{payload.doubt_text}\n\n"
        f"ID: {doubt_id}"
    )

    # background send + update notify_status
    async def _notify_and_update():
        success = await send_telegram_message(TELEGRAM_BOT_TOKEN, TEACHER_CHAT_ID, text)
        new_status = "success" if success else "failed"
        try:
            await db.doubts.update_one({"_id": res.inserted_id}, {"$set": {"notify_status": new_status}})
        except Exception:
            logger.exception("Failed to update notify status for %s", doubt_id)

    background_tasks.add_task(_notify_and_update)
    return {"id": doubt_id, "status": "queued"}

@app.get("/doubts", response_model=List[Dict[str, Any]])
async def list_doubts(status: Optional[str] = None, limit: int = 50, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    query = {}
    if status:
        query["status"] = status
    cursor = db.doubts.find(query).sort("created_at", -1).limit(min(limit, 200))
    docs = []
    async for d in cursor:
        docs.append(serialize_doc(d))
    return docs

@app.get("/doubts/{doubt_id}", response_model=Dict[str, Any])
async def get_doubt(doubt_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        oid = ObjectId(doubt_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    doc = await db.doubts.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="not-found")
    return serialize_doc(doc)

class DoubtUpdate(BaseModel):
    status: Optional[str] = None
    teacher_reply: Optional[str] = None

@app.patch("/doubts/{doubt_id}", response_model=Dict[str, Any])
async def update_doubt(doubt_id: str, payload: DoubtUpdate, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        oid = ObjectId(doubt_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    update_doc = {}
    if payload.status:
        update_doc["status"] = payload.status
    if payload.teacher_reply:
        update_doc["teacher_reply"] = payload.teacher_reply
    if not update_doc:
        raise HTTPException(status_code=400, detail="nothing-to-update")
    await db.doubts.update_one({"_id": oid}, {"$set": update_doc})
    doc = await db.doubts.find_one({"_id": oid})
    return serialize_doc(doc)

# ---------- Simple ping endpoint for Telegram test ----------
@app.post("/internal/test-telegram")
async def test_telegram(db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    txt = f"Test message from backend at {datetime.utcnow().isoformat()}"
    ok = await send_telegram_message(TELEGRAM_BOT_TOKEN, TEACHER_CHAT_ID, txt)
    return {"ok": ok}

# ---------- Basic root ----------
@app.get("/")
async def root():
    return {"service": "zynno-backend", "uptime": datetime.utcnow().isoformat()}

# ---------- Run with: uvicorn main:app --host 0.0.0.0 --port 8000 ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)

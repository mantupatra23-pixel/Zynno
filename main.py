# main.py (FINAL — Doubt saving + Teacher dashboard + Rate limit + Auth + Stats)
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------- ENV ----------
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME") or "zynno"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TEACHER_CHAT_ID = os.getenv("TEACHER_CHAT_ID")
TEACHER_API_KEY = os.getenv("TEACHER_API_KEY")  # required to protect teacher endpoints
RATE_LIMIT_MAX_PER_HOUR = int(os.getenv("RATE_LIMIT_MAX_PER_HOUR") or 6)  # default 6 doubts/hour per student/ip

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zynno")

# ---------- App ----------
app = FastAPI(title="Zynno - Final Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB ----------
mongo_client: Optional[AsyncIOMotorClient] = None
db = None

async def get_db():
    return db

# ---------- Models ----------
# limit lengths to avoid abuse
ShortName = constr(min_length=1, max_length=60)
ShortClass = constr(min_length=1, max_length=10)
ShortDoubt = constr(min_length=3, max_length=2000)

class SubmitDoubtIn(BaseModel):
    student_name: ShortName
    student_class: ShortClass
    doubt_text: ShortDoubt
    student_telegram_id: Optional[str] = None  # optional for later student notify

class DoubtOut(BaseModel):
    id: str
    student_name: str
    student_class: str
    doubt_text: str
    created_at: datetime
    status: str
    teacher_reply: Optional[str] = None
    notify_status: Optional[str] = None

class DoubtUpdate(BaseModel):
    status: Optional[str] = None
    teacher_reply: Optional[str] = None

# ---------- Rate limiter (per student or ip) ----------
RATE_WINDOW = timedelta(hours=1)
_rate_store: Dict[str, List[datetime]] = {}

def _now():
    return datetime.utcnow()

def check_and_add_rate(key: str) -> bool:
    now = _now()
    arr = _rate_store.get(key, [])
    arr = [t for t in arr if now - t <= RATE_WINDOW]
    if len(arr) >= RATE_LIMIT_MAX_PER_HOUR:
        _rate_store[key] = arr
        return False
    arr.append(now)
    _rate_store[key] = arr
    _rate_store[key] = arr
    return True

# ---------- Helpers ----------
def serialize_doc(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    _id = out.pop("_id", None)
    out["id"] = str(_id) if _id else None
    if isinstance(out.get("created_at"), datetime):
        out["created_at"] = out["created_at"].isoformat()
    return out

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
                    logger.info("Telegram message sent")
                    return True
                else:
                    logger.warning("Telegram failed status=%s body=%s", resp.status_code, resp.text)
            except Exception as e:
                logger.exception("Telegram exception: %s", e)
            await asyncio.sleep(1 * attempt)
    return False

# ---------- Auth ----------
def require_teacher_api_key(x_api_key: Optional[str] = Header(None)):
    if not TEACHER_API_KEY:
        logger.warning("TEACHER_API_KEY not configured; teacher endpoints are OPEN")
        return
    if not x_api_key or x_api_key != TEACHER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ---------- Startup / Shutdown ----------
@app.on_event("startup")
async def startup():
    global mongo_client, db
    if MONGO_URI:
        mongo_client = AsyncIOMotorClient(MONGO_URI)
        db = mongo_client[MONGO_DB_NAME]
        # ensure indexes
        try:
            await db.doubts.create_index([("status", 1)])
            await db.doubts.create_index([("created_at", -1)])
            await db.doubts.create_index([("student_class", 1)])
            await db.doubts.create_index([("student_telegram_id", 1)])
            logger.info("MongoDB connected and indexes ensured")
        except Exception as e:
            logger.exception("Index creation failed: %s", e)
    else:
        logger.warning("MONGO_URI not provided. DB features disabled.")

@app.on_event("shutdown")
async def shutdown():
    global mongo_client
    if mongo_client:
        mongo_client.close()

# ---------- Health ----------
@app.get("/health")
async def health():
    info = {"status": "ok", "db": False}
    try:
        if db:
            await db.command("ping")
            info["db"] = True
    except Exception as e:
        logger.warning("Health DB ping failed: %s", e)
    return info

# ---------- Submit doubt (student facing) ----------
@app.post("/submit_doubt", response_model=Dict[str, Any])
async def submit_doubt(request: Request, payload: SubmitDoubtIn, background_tasks: BackgroundTasks, db=Depends(get_db)):
    # rate limiting: prefer student_telegram_id if present else client ip
    client_ip = request.client.host if request.client else "unknown"
    rate_key = payload.student_telegram_id or (payload.student_name + "|" + client_ip)
    if not check_and_add_rate(rate_key):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({RATE_LIMIT_MAX_PER_HOUR}/hour)")

    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")

    doc = {
        "student_name": payload.student_name,
        "student_class": payload.student_class,
        "doubt_text": payload.doubt_text,
        "student_telegram_id": payload.student_telegram_id,
        "status": "open",
        "teacher_reply": None,
        "created_at": datetime.utcnow(),
        "notify_status": "queued"
    }
    res = await db.doubts.insert_one(doc)
    doubt_id = str(res.inserted_id)

    text = (
        f"<b>📚 New Doubt</b>\n\n"
        f"<b>Student:</b> {payload.student_name}\n"
        f"<b>Class:</b> {payload.student_class}\n\n"
        f"{payload.doubt_text}\n\n"
        f"ID: {doubt_id}"
    )

    async def _notify_and_update():
        ok = await send_telegram_message(TELEGRAM_BOT_TOKEN, TEACHER_CHAT_ID, text)
        status = "success" if ok else "failed"
        try:
            await db.doubts.update_one({"_id": res.inserted_id}, {"$set": {"notify_status": status}})
        except Exception:
            logger.exception("Failed to update notify_status")

    background_tasks.add_task(_notify_and_update)

    return {"id": doubt_id, "status": "queued"}

# ---------- Teacher: list doubts ----------
@app.get("/doubts", response_model=List[Dict[str, Any]])
async def list_doubts(status: Optional[str] = None, klass: Optional[str] = None, limit: int = 100, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    query = {}
    if status:
        query["status"] = status
    if klass:
        query["student_class"] = klass
    cursor = db.doubts.find(query).sort("created_at", -1).limit(min(limit, 500))
    out = []
    async for d in cursor:
        out.append(serialize_doc(d))
    return out

# ---------- Teacher: get a doubt ----------
@app.get("/doubts/{doubt_id}", response_model=Dict[str, Any])
async def get_doubt(doubt_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        oid = ObjectId(doubt_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    d = await db.doubts.find_one({"_id": oid})
    if not d:
        raise HTTPException(status_code=404, detail="not-found")
    return serialize_doc(d)

# ---------- Teacher: reply to doubt (saves reply, marks answered, optionally message student) ----------
@app.post("/reply/{doubt_id}", response_model=Dict[str, Any])
async def reply_to_doubt(doubt_id: str, payload: DoubtUpdate, background_tasks: BackgroundTasks, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        oid = ObjectId(doubt_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    update_doc = {}
    if payload.teacher_reply:
        update_doc["teacher_reply"] = payload.teacher_reply
    if payload.status:
        update_doc["status"] = payload.status
    if not update_doc:
        raise HTTPException(status_code=400, detail="nothing-to-update")
    await db.doubts.update_one({"_id": oid}, {"$set": update_doc})
    # fetch doc to see student_telegram_id
    doc = await db.doubts.find_one({"_id": oid})
    # if student_telegram_id present, notify student (best-effort)
    student_chat = doc.get("student_telegram_id")
    if student_chat and payload.teacher_reply:
        text = f"📚 <b>Answer to your doubt</b>\n\n{payload.teacher_reply}\n\n(From teacher)"
        # send in background
        async def _notify_student():
            await send_telegram_message(TELEGRAM_BOT_TOKEN, student_chat, text)
        background_tasks.add_task(_notify_student)
    return serialize_doc(doc)

# ---------- Teacher: update status only ----------
@app.patch("/doubts/{doubt_id}/status", response_model=Dict[str, Any])
async def update_status(doubt_id: str, status_body: Dict[str, str], db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    s = status_body.get("status")
    if not s:
        raise HTTPException(status_code=400, detail="missing-status")
    try:
        oid = ObjectId(doubt_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    await db.doubts.update_one({"_id": oid}, {"$set": {"status": s}})
    doc = await db.doubts.find_one({"_id": oid})
    return serialize_doc(doc)

# ---------- Stats / analytics ----------
@app.get("/stats/overview", response_model=Dict[str, Any])
async def stats_overview(db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    # total counts
    total = await db.doubts.count_documents({})
    open_count = await db.doubts.count_documents({"status": "open"})
    answered_count = await db.doubts.count_documents({"status": "answered"})
    # last 24h
    since = datetime.utcnow() - timedelta(hours=24)
    last24 = await db.doubts.count_documents({"created_at": {"$gte": since}})
    return {
        "total_doubts": total,
        "open": open_count,
        "answered": answered_count,
        "last_24h": last24,
        "rate_limit_per_hour": RATE_LIMIT_MAX_PER_HOUR
    }

# ---------- Test telegram send ----------
@app.post("/internal/test-telegram")
async def test_telegram(db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    txt = f"Test message from backend at {datetime.utcnow().isoformat()}"
    ok = await send_telegram_message(TELEGRAM_BOT_TOKEN, TEACHER_CHAT_ID, txt)
    return {"ok": ok}

# ---------- root ----------
@app.get("/")
async def root():
    return {"service": "zynno-backend", "time": datetime.utcnow().isoformat()}

# ---------------------------
# Student Profile & Chapter System
# ---------------------------
from typing import Any

# Pydantic models for Student & Chapter
class StudentIn(BaseModel):
    name: ShortName
    klass: ShortClass
    board: Optional[str] = "CBSE"
    phone: Optional[str] = None
    telegram_id: Optional[str] = None

class StudentOut(StudentIn):
    id: str
    created_at: datetime
    streak: Optional[int] = 0

class ChapterIn(BaseModel):
    title: str
    subject: str
    class_for: ShortClass
    order: Optional[int] = 0     # order within subject/class
    notes_url: Optional[str] = None
    video_url: Optional[str] = None
    pdf_url: Optional[str] = None
    tags: Optional[List[str]] = []

class ChapterOut(ChapterIn):
    id: str
    created_at: datetime
    published: bool = True
    is_premium: bool = False

# ---------- Student endpoints ----------
@app.post("/student/register", response_model=Dict[str, Any])
async def student_register(payload: StudentIn, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow(), "streak": 0, "completed_chapters": []})
    # upsert by telegram_id if available
    if payload.telegram_id:
        res = await db.students.update_one({"telegram_id": payload.telegram_id}, {"$set": doc}, upsert=True)
        if res.upserted_id:
            sid = str(res.upserted_id)
        else:
            row = await db.students.find_one({"telegram_id": payload.telegram_id})
            sid = str(row["_id"])
        return {"ok": True, "id": sid}
    res = await db.students.insert_one(doc)
    return {"ok": True, "id": str(res.inserted_id)}

@app.get("/student/{student_id}", response_model=Dict[str, Any])
async def get_student(student_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        oid = ObjectId(student_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    row = await db.students.find_one({"_id": oid})
    if not row:
        raise HTTPException(status_code=404, detail="not-found")
    return serialize_doc(row)

# mark chapter complete for a student
@app.post("/student/{student_id}/complete_chapter/{chapter_id}", response_model=Dict[str, Any])
async def complete_chapter(student_id: str, chapter_id: str, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        sid = ObjectId(student_id)
        cid = ObjectId(chapter_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    # add chapter to student's completed list if not present
    await db.students.update_one({"_id": sid}, {"$addToSet": {"completed_chapters": chapter_id}})
    # optional: update streak / progress metrics (simple increment)
    await db.students.update_one({"_id": sid}, {"$inc": {"streak": 1}})
    return {"ok": True, "student_id": student_id, "chapter_id": chapter_id}

# ---------- Chapter Admin endpoints (teacher protected) ----------
@app.post("/admin/create_chapter", response_model=Dict[str, Any])
async def admin_create_chapter(payload: ChapterIn, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow(), "published": True})
    res = await db.chapters.insert_one(doc)
    return {"ok": True, "id": str(res.inserted_id)}

@app.get("/chapters", response_model=List[Dict[str, Any]])
async def list_chapters(class_for: Optional[str] = None, subject: Optional[str] = None, limit: int = 200, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    query = {}
    if class_for:
        query["class_for"] = class_for
    if subject:
        query["subject"] = subject
    cursor = db.chapters.find(query).sort([("class_for", 1), ("order", 1)]).limit(min(limit, 1000))
    out = []
    async for d in cursor:
        out.append(serialize_doc(d))
    return out

@app.get("/chapters/{chapter_id}", response_model=Dict[str, Any])
async def get_chapter(chapter_id: str, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        cid = ObjectId(chapter_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    doc = await db.chapters.find_one({"_id": cid})
    if not doc:
        raise HTTPException(status_code=404, detail="not-found")
    return serialize_doc(doc)

@app.patch("/admin/chapter/{chapter_id}", response_model=Dict[str, Any])
async def update_chapter(chapter_id: str, payload: ChapterIn, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        cid = ObjectId(chapter_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    update_doc = payload.dict()
    await db.chapters.update_one({"_id": cid}, {"$set": update_doc})
    doc = await db.chapters.find_one({"_id": cid})
    return serialize_doc(doc)

# ---------- Student: Today's Chapter logic ----------
@app.get("/student/today/{student_id}", response_model=Dict[str, Any])
async def student_today(student_id: str, db=Depends(get_db)):
    """
    Simple logic:
    - load student
    - find all chapters for student's class sorted by order
    - find first chapter not in student's completed_chapters
    - return that as today's chapter
    """
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        sid = ObjectId(student_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    student = await db.students.find_one({"_id": sid})
    if not student:
        raise HTTPException(status_code=404, detail="student-not-found")
    klass = student.get("klass")
    completed = student.get("completed_chapters", [])
    cursor = db.chapters.find({"class_for": klass, "published": True}).sort("order", 1)
    async for ch in cursor:
        cid_str = str(ch["_id"])
        if cid_str not in completed:
            # return this chapter as today's
            ch_out = serialize_doc(ch)
            return {"today": ch_out, "remaining_in_class": await db.chapters.count_documents({"class_for": klass, "published": True}) - len(completed)}
    # all done
    return {"today": None, "message": "All chapters completed for this class."}

# ---------- run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)

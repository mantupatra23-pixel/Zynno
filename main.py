from __future__ import annotations

# -------- CLEAN IMPORT BLOCK --------
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# FastAPI core
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks, Header, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Models (your custom models)
from models.user import User
from models.student import Student
from models.classroom import Classroom

# Pydantic base
from pydantic import BaseModel, Field, constr

# MongoDB driver
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# External HTTP client
import httpx

# OCR / Image processing
import pytesseract
from PIL import Image

# Env loader
from dotenv import load_dotenv
load_dotenv()

# ---------- ENV ----------
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME") or "zynno"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TEACHER_CHAT_ID = os.getenv("TEACHER_CHAT_ID")
TEACHER_API_KEY = os.getenv("TEACHER_API_KEY")
RATE_LIMIT_MAX_PER_HOUR = int(os.getenv("RATE_LIMIT_MAX_PER_HOUR", "60"))

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

# ---------- Health -----------
@app.get("/health")
async def health():
    """
    Health endpoint returns {"status":"ok","db": True/False}
    Uses mongo_client.admin.command("ping") to check DB connectivity.
    """
    info = {"status": "ok", "db": False}
    try:
        # make sure mongo_client exists and is connected
        if mongo_client is None:
            # no client configured
            info["db"] = False
        else:
            # use the admin database ping command (recommended)
            await mongo_client.admin.command("ping")
            info["db"] = True
    except Exception as e:
        # log the exception (logger must be defined earlier in your file)
        try:
            logger.warning("Health DB ping failed: %s", e)
        except Exception:
            pass
        info["db"] = False

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

# ---------------------------
# TEST MODULE (MCQ engine)
# ---------------------------
from typing import Tuple

# Pydantic models for Test & Attempts
class MCQItem(BaseModel):
    q: str
    options: List[str]
    answer_index: int  # correct option index (0-based)
    marks: Optional[int] = 1

class TestIn(BaseModel):
    title: str
    subject: str
    class_for: ShortClass
    questions: List[MCQItem]
    time_limit_minutes: Optional[int] = 30  # optional time limit
    is_published: bool = True
    is_paid: bool = False

class TestOut(BaseModel):
    id: str
    title: str
    subject: str
    class_for: str
    question_count: int
    time_limit_minutes: Optional[int]

class AttemptIn(BaseModel):
    student_name: str
    student_class: ShortClass
    student_telegram_id: Optional[str] = None

class SubmitAnswersIn(BaseModel):
    answers: List[int]  # indexes chosen by student in order of questions

# Admin: create test
@app.post("/admin/create_test", response_model=Dict[str, Any])
async def admin_create_test(payload: TestIn, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    doc = payload.dict()
    # store questions but DO NOT expose answers in public endpoints
    res = await db.tests.insert_one({
        "title": doc["title"],
        "subject": doc["subject"],
        "class_for": doc["class_for"],
        "questions": doc["questions"],  # stored with correct answers
        "time_limit_minutes": doc.get("time_limit_minutes", 30),
        "is_published": doc.get("is_published", True),
        "is_paid": doc.get("is_published", False),
        "created_at": datetime.utcnow()
    })
    return {"ok": True, "id": str(res.inserted_id)}

# Public: list tests (no answers returned)
@app.get("/tests", response_model=List[Dict[str, Any]])
async def list_tests(class_for: Optional[str] = None, subject: Optional[str] = None, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    q = {}
    if class_for: q["class_for"] = class_for
    if subject: q["subject"] = subject
    cursor = db.tests.find(q).sort("created_at", -1)
    out = []
    async for t in cursor:
        t_obj = {
            "id": str(t["_id"]),
            "title": t["title"],
            "subject": t["subject"],
            "class_for": t["class_for"],
            "question_count": len(t.get("questions", [])),
            "time_limit_minutes": t.get("time_limit_minutes", 30),
            "is_published": t.get("is_published", True)
        }
        out.append(t_obj)
    return out

# Public: get test (without answers)
@app.get("/tests/{test_id}", response_model=Dict[str, Any])
async def get_test_meta(test_id: str, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        tid = ObjectId(test_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    t = await db.tests.find_one({"_id": tid})
    if not t:
        raise HTTPException(status_code=404, detail="not-found")
    # remove correct answers before returning
    questions = []
    for q in t.get("questions", []):
        questions.append({"q": q["q"], "options": q["options"], "marks": q.get("marks", 1)})
    return {
        "id": str(t["_id"]),
        "title": t["title"],
        "subject": t["subject"],
        "class_for": t["class_for"],
        "questions": questions,
        "time_limit_minutes": t.get("time_limit_minutes", 30)
    }

# Student: start attempt -> creates attempt doc and returns attempt_id + start_time + expiry
@app.post("/test/start/{test_id}", response_model=Dict[str, Any])
async def start_test(test_id: str, payload: AttemptIn, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        tid = ObjectId(test_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    test = await db.tests.find_one({"_id": tid, "is_published": True})
    if not test:
        raise HTTPException(status_code=404, detail="test-not-found")
    start = datetime.utcnow()
    expiry = start + timedelta(minutes=test.get("time_limit_minutes", 30))
    attempt_doc = {
        "test_id": test_id,
        "student_name": payload.student_name,
        "student_class": payload.student_class,
        "student_telegram_id": payload.student_telegram_id,
        "start_time": start,
        "expiry_time": expiry,
        "answers": [],  # to be filled
        "score": None,
        "max_score": sum([q.get("marks",1) for q in test.get("questions", [])]),
        "status": "in_progress",  # in_progress / submitted / graded
        "created_at": datetime.utcnow()
    }
    res = await db.attempts.insert_one(attempt_doc)
    return {"attempt_id": str(res.inserted_id), "start_time": start.isoformat(), "expiry_time": expiry.isoformat()}

# Helper: grade MCQ (returns (score, detail_list))
def grade_mcq(questions: List[dict], answers: List[int]) -> Tuple[int, List[dict]]:
    score = 0
    details = []
    for idx, q in enumerate(questions):
        correct_idx = int(q.get("answer_index", 0))
        marks = int(q.get("marks", 1))
        chosen = answers[idx] if idx < len(answers) else None
        correct = (chosen is not None and chosen == correct_idx)
        if correct:
            score += marks
        details.append({"q_index": idx, "chosen": chosen, "correct_index": correct_idx, "marks": marks, "correct": correct})
    return score, details

# Student: submit answers (by attempt id) -> auto-grade
@app.post("/test/submit/{attempt_id}", response_model=Dict[str, Any])
async def submit_test(attempt_id: str, payload: SubmitAnswersIn, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        aid = ObjectId(attempt_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    attempt = await db.attempts.find_one({"_id": aid})
    if not attempt:
        raise HTTPException(status_code=404, detail="attempt-not-found")
    if attempt.get("status") != "in_progress":
        raise HTTPException(status_code=400, detail="attempt-already-submitted")
    # fetch test
    test = await db.tests.find_one({"_id": ObjectId(attempt["test_id"])})
    if not test:
        raise HTTPException(status_code=404, detail="test-not-found")
    questions = test.get("questions", [])
    # optional: check expiry
    if datetime.utcnow() > attempt.get("expiry_time"):
        # mark as submitted but score 0 (or allow grading partial)
        await db.attempts.update_one({"_id": aid}, {"$set": {"status": "submitted", "answers": payload.answers, "score": 0, "graded_at": datetime.utcnow()}})
        return {"ok": True, "status": "expired", "score": 0}
    # grade
    score, details = grade_mcq(questions, payload.answers)
    await db.attempts.update_one({"_id": aid}, {"$set": {"status": "submitted", "answers": payload.answers, "score": score, "graded_at": datetime.utcnow(), "grading_details": details}})
    # update test-level analytics: increment attempts, total score
    try:
        await db.tests.update_one({"_id": ObjectId(attempt["test_id"])}, {"$inc": {"attempts_count": 1, "total_score": score}})
    except Exception:
        logger.exception("Failed to update test analytics")
    return {"ok": True, "score": score, "max_score": attempt.get("max_score"), "details": details}

# Teacher: list attempts for a test
@app.get("/admin/test/{test_id}/attempts", response_model=List[Dict[str, Any]])
async def list_attempts(test_id: str, limit: int = 100, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        tid = ObjectId(test_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    cursor = db.attempts.find({"test_id": str(tid)}).sort("created_at", -1).limit(min(limit,500))
    out = []
    async for a in cursor:
        out.append(serialize_doc(a))
    return out

# Teacher: get single attempt detail
@app.get("/admin/attempt/{attempt_id}", response_model=Dict[str, Any])
async def get_attempt(attempt_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        aid = ObjectId(attempt_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    a = await db.attempts.find_one({"_id": aid})
    if not a:
        raise HTTPException(status_code=404, detail="not-found")
    return serialize_doc(a)

# Test analytics endpoint (avg score)
@app.get("/admin/test/{test_id}/analytics", response_model=Dict[str, Any])
async def test_analytics(test_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        tid = ObjectId(test_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    test = await db.tests.find_one({"_id": tid})
    if not test:
        raise HTTPException(status_code=404, detail="not-found")
    attempts_count = test.get("attempts_count", 0)
    total_score = test.get("total_score", 0)
    avg = (total_score / attempts_count) if attempts_count > 0 else 0
    return {"attempts": attempts_count, "total_score": total_score, "average_score": avg}

# ---------------------------
# HOMEWORK MODULE (assign, submit, review, grade)
# ---------------------------
from fastapi import UploadFile, File
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
import io
from PIL import Image

# import pytesseract

# ensure gridfs bucket available on startup if db present
gridfs_bucket = None
@app.on_event("startup")
async def setup_gridfs():
    global gridfs_bucket, db
    if db:
        try:
            gridfs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="homework_files")
            logger.info("GridFS bucket ready")
        except Exception as e:
            logger.exception("GridFS setup failed: %s", e)

# Pydantic models
class HomeworkIn(BaseModel):
    title: str
    chapter_id: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[datetime] = None  # ISO format string accepted

class HomeworkOut(HomeworkIn):
    id: str
    created_at: datetime
    assigned_by: Optional[str] = None

class HomeworkSubmitResponse(BaseModel):
    id: str
    homework_id: str
    student_name: str
    file_id: Optional[str] = None
    ocr_text: Optional[str] = None
    created_at: datetime
    status: str

# Admin: assign homework
@app.post("/admin/assign_homework", response_model=Dict[str, Any])
async def assign_homework(payload: HomeworkIn, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow(), "assigned_by": TEACHER_CHAT_ID or "teacher", "published": True})
    res = await db.homeworks.insert_one(doc)
    return {"ok": True, "id": str(res.inserted_id)}

# Public: list homeworks (student view)
@app.get("/homeworks", response_model=List[Dict[str, Any]])
async def list_homeworks(class_for: Optional[str] = None, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    q = {}
    if class_for:
        q["class_for"] = class_for
    cursor = db.homeworks.find(q).sort("created_at", -1)
    out = []
    async for h in cursor:
        out.append(serialize_doc(h))
    return out

# Student: submit homework (file upload)
# Fields: student_name, student_class (as form fields) + file
@app.post("/student/submit_homework/{homework_id}", response_model=Dict[str, Any])
async def submit_homework(homework_id: str, student_name: str = Form(...), student_class: str = Form(...),
                          file: UploadFile = File(...), background_tasks: BackgroundTasks = None, db=Depends(get_db)):
    """
    Multipart form:
    - student_name (form)
    - student_class (form)
    - file (upload)
    """
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        hw_oid = ObjectId(homework_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-homework-id")
    # save file into GridFS
    file_bytes = await file.read()
    file_name = file.filename or f"upload_{datetime.utcnow().timestamp()}"
    file_id = None
    try:
        if gridfs_bucket:
            stream = io.BytesIO(file_bytes)
            file_id = await gridfs_bucket.upload_from_stream(file_name, stream)
        else:
            # fallback: store in a collection (base64) — but we store metadata only
            file_id = None
    except Exception as e:
        logger.exception("GridFS upload failed: %s", e)
        file_id = None

    # attempt OCR (best-effort)
    ocr_text = None
    try:
        # try open as image
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        ocr_text = pytesseract.image_to_string(img)
    except Exception as e:
        # OCR failed (likely tesseract binary missing), skip silently and log
        logger.warning("OCR attempt failed or skipped: %s", e)
        ocr_text = None

    doc = {
        "homework_id": homework_id,
        "student_name": student_name,
        "student_class": student_class,
        "file_id": str(file_id) if file_id else None,
        "file_name": file_name,
        "ocr_text": ocr_text,
        "status": "submitted",
        "created_at": datetime.utcnow(),
        "graded": False,
        "grade": None,
        "feedback": None
    }
    res = await db.homework_submissions.insert_one(doc)
    submission_id = str(res.inserted_id)

    # notify teacher in background
    text = (f"📥 New Homework Submission\nStudent: {student_name}\nClass: {student_class}\nHomework ID: {homework_id}\nSubmission ID: {submission_id}")
    if background_tasks:
        background_tasks.add_task(asyncio.create_task, send_telegram_message(TELEGRAM_BOT_TOKEN, TEACHER_CHAT_ID, text))
    else:
        # try fire-and-forget
        asyncio.create_task(send_telegram_message(TELEGRAM_BOT_TOKEN, TEACHER_CHAT_ID, text))

    return {"ok": True, "submission_id": submission_id, "file_id": str(file_id) if file_id else None, "ocr_text_preview": (ocr_text[:200] if ocr_text else None)}

# Admin: list submissions for a homework
@app.get("/admin/homework/{homework_id}/submissions", response_model=List[Dict[str, Any]])
async def list_homework_submissions(homework_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        hw_oid = ObjectId(homework_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-homework-id")
    cursor = db.homework_submissions.find({"homework_id": homework_id}).sort("created_at", -1)
    out = []
    async for s in cursor:
        out.append(serialize_doc(s))
    return out

# Admin: get single submission (metadata + OCR text)
@app.get("/admin/homework/submission/{submission_id}", response_model=Dict[str, Any])
async def get_submission(submission_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        sid = ObjectId(submission_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    doc = await db.homework_submissions.find_one({"_id": sid})
    if not doc:
        raise HTTPException(status_code=404, detail="not-found")
    return serialize_doc(doc)

# Admin: grade a submission (add grade + feedback)
@app.post("/admin/homework/grade/{submission_id}", response_model=Dict[str, Any])
async def grade_submission(submission_id: str, payload: Dict[str, Any], db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    """
    payload: {"grade": 8, "feedback": "Good work", "status": "graded"}
    """
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        sid = ObjectId(submission_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    update = {}
    if "grade" in payload:
        update["grade"] = payload["grade"]
        update["graded"] = True
    if "feedback" in payload:
        update["feedback"] = payload["feedback"]
    if "status" in payload:
        update["status"] = payload["status"]
    if not update:
        raise HTTPException(status_code=400, detail="nothing-to-update")
    await db.homework_submissions.update_one({"_id": sid}, {"$set": update})
    doc = await db.homework_submissions.find_one({"_id": sid})
    # optionally notify student via telegram if student_telegram_id present
    student_chat = doc.get("student_telegram_id")
    if student_chat and ("feedback" in update or "grade" in update):
        text = f"📣 Your homework has been graded.\nGrade: {update.get('grade')}\nFeedback: {update.get('feedback','')}"
        asyncio.create_task(send_telegram_message(TELEGRAM_BOT_TOKEN, student_chat, text))
    return serialize_doc(doc)

# ---------------------------
# LIVE CLASS SYSTEM (Agora Ready)
# ---------------------------

class LiveClassIn(BaseModel):
    title: str
    subject: str
    class_for: ShortClass
    scheduled_at: datetime
    duration_minutes: int = 45
    teacher_name: Optional[str] = "Teacher"
    description: Optional[str] = None

class LiveClassOut(BaseModel):
    id: str
    title: str
    subject: str
    class_for: str
    scheduled_at: datetime
    duration_minutes: int
    teacher_name: str
    description: Optional[str]
    is_live: bool
    agora_channel: str
    agora_token_teacher: Optional[str] = None
    agora_token_student: Optional[str] = None


# Admin: Create Live Class
@app.post("/admin/live/create", response_model=Dict[str, Any])
async def create_live_class(payload: LiveClassIn, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(503, "db-not-configured")

    class_doc = payload.dict()
    class_doc.update({
        "created_at": datetime.utcnow(),
        "is_live": False,
        "agora_channel": f"class_{int(datetime.utcnow().timestamp())}"
    })

    res = await db.live_classes.insert_one(class_doc)
    return {"ok": True, "id": str(res.inserted_id), "agora_channel": class_doc["agora_channel"]}


# Public: List Live Classes
@app.get("/live/classes", response_model=List[Dict[str, Any]])
async def list_live_classes(class_for: Optional[str] = None, db=Depends(get_db)):
    if not db:
        raise HTTPException(503, "db-not-configured")

    q = {}
    if class_for:
        q["class_for"] = class_for

    cursor = db.live_classes.find(q).sort("scheduled_at", 1)
    out = []
    async for c in cursor:
        out.append(serialize_doc(c))
    return out


# Teacher: Start Live Class (Generate Agora token)
@app.post("/admin/live/start/{class_id}", response_model=Dict[str, Any])
async def start_live_class(class_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(503, "db-not-configured")

    try:
        cid = ObjectId(class_id)
    except:
        raise HTTPException(400, "invalid-id")

    live = await db.live_classes.find_one({"_id": cid})
    if not live:
        raise HTTPException(404, "not-found")

    channel = live["agora_channel"]

    # Agora token dummy placeholders (client will generate real token)
    teacher_token = f"TEACHER_TOKEN_{channel}"
    student_token = f"STUDENT_TOKEN_{channel}"

    await db.live_classes.update_one(
        {"_id": cid},
        {"$set": {"is_live": True, "start_time": datetime.utcnow(),
                  "agora_teacher_token": teacher_token,
                  "agora_student_token": student_token}}
    )

    return {
        "ok": True,
        "channel": channel,
        "teacher_token": teacher_token,
        "student_token": student_token
    }


# Student: Join live class
@app.get("/live/join/{class_id}", response_model=Dict[str, Any])
async def join_live_class(class_id: str, student_name: Optional[str] = None, db=Depends(get_db)):
    if not db:
        raise HTTPException(503, "db-not-configured")

    try:
        cid = ObjectId(class_id)
    except:
        raise HTTPException(400, "invalid-id")

    live = await db.live_classes.find_one({"_id": cid})
    if not live:
        raise HTTPException(404, "not-found")

    if live.get("is_live") is False:
        return {"live": False, "message": "Class has not started yet."}

    # mark attendance
    if student_name:
        await db.live_attendance.insert_one({
            "class_id": class_id,
            "student_name": student_name,
            "joined_at": datetime.utcnow()
        })

    return {
        "live": True,
        "channel": live["agora_channel"],
        "token": live.get("agora_student_token")
    }


# Teacher: End class
@app.post("/admin/live/end/{class_id}", response_model=Dict[str, Any])
async def end_live_class(class_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(503, "db-not-configured")

    try:
        cid = ObjectId(class_id)
    except:
        raise HTTPException(400, "invalid-id")

    await db.live_classes.update_one(
        {"_id": cid},
        {"$set": {"is_live": False, "end_time": datetime.utcnow()}}
    )

    return {"ok": True, "ended": True}

# ---------------------------
# AI AUTO-TEACHER ENGINE
# ---------------------------

import base64
import json
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")  # default

if not OPENAI_API_KEY:
    logger.warning("AI features disabled (no OPENAI_API_KEY)")


# Helper to call AI model
async def ai_generate(prompt: str) -> str:
    if not OPENAI_API_KEY:
        return "AI key missing"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    data = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    async with httpx.AsyncClient(timeout=40) as client:
        r = await client.post(url, json=data, headers=headers)
        res = r.json()
        return res["choices"][0]["message"]["content"]


# AI: Explain chapter
@app.post("/ai/explain")
async def ai_explain(data: Dict[str, str]):
    chapter = data.get("chapter")
    subject = data.get("subject")
    klass = data.get("class")
    
    prompt = f"""
    Explain the chapter '{chapter}' of class {klass}, subject {subject}, 
    in very simple language, step-by-step, with examples and diagrams explained in words.
    """
    ans = await ai_generate(prompt)
    return {"explanation": ans}


# AI: Notes generator
@app.post("/ai/notes")
async def ai_notes(data: Dict[str, str]):
    chapter = data.get("chapter")
    subject = data.get("subject")

    prompt = f"""
    Create complete study notes for chapter '{chapter}' (subject: {subject}).
    Include:
    - Important definitions
    - Key formulas
    - Examples
    - Key points
    - Mindmap-style breakdown
    - Tips to remember
    """
    ans = await ai_generate(prompt)
    return {"notes": ans}


# AI: Quiz generator
@app.post("/ai/quiz")
async def ai_quiz(data: Dict[str, str]):
    chapter = data.get("chapter")
    count = int(data.get("count", 5))

    prompt = f"""
    Create {count} MCQ questions for chapter '{chapter}'.
    Output JSON:
    [
      {{"question": "...", "options": ["A","B","C","D"], "answer": "A"}}
    ]
    """
    ans = await ai_generate(prompt)
    try:
        parsed = json.loads(ans)
    except:
        parsed = [{"error": "AI output not JSON", "raw": ans}]
    return {"quiz": parsed}


# AI: Summary
@app.post("/ai/summary")
async def ai_summary(data: Dict[str, str]):
    chapter = data.get("chapter")

    prompt = f"""
    Summarize chapter '{chapter}' in 10 easy bullet points for quick revision.
    """
    ans = await ai_generate(prompt)
    return {"summary": ans}


# AI: Voice generator (TTS)
@app.post("/ai/voice")
async def ai_voice(data: Dict[str, str]):
    text = data.get("text", "")

    if not text:
        raise HTTPException(400, "text required")

    url = "https://api.openai.com/v1/audio/speech"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    body = {
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "input": text
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=body)
        audio_bytes = response.content

    b64 = base64.b64encode(audio_bytes).decode()
    return {"audio_base64": b64}

# ---------------------------
# LEADERBOARD & RANKING SYSTEM
# ---------------------------
from pymongo import ASCENDING, DESCENDING

# weights (you can change)
WEIGHT_TEST_SCORE = 1.0      # each scored point from tests
WEIGHT_HOMEWORK = 2.0        # each graded homework point multiplier
WEIGHT_DOUBT_SUBMIT = 0.2    # reward for asking doubts (engagement)
WEIGHT_ACTIVITY = 0.1        # miscellaneous activity bonus

# ensure indexes for leaderboard aggregation on startup
@app.on_event("startup")
async def ensure_leaderboard_indexes():
    global db
    if db:
        try:
            await db.attempts.create_index([("student_telegram_id", ASCENDING)])
            await db.attempts.create_index([("student_name", ASCENDING)])
            await db.homework_submissions.create_index([("student_telegram_id", ASCENDING)])
            await db.homework_submissions.create_index([("student_name", ASCENDING)])
            await db.leaderboard.create_index([("snapshot_name", ASCENDING)])
            logger.info("Leaderboard indexes ensured")
        except Exception:
            logger.exception("Failed to create leaderboard indexes")

# Helper: compute scores aggregation pipeline
async def _compute_scores_by_student(class_for: Optional[str] = None):
    """
    Aggregate across attempts (tests) and homework_submissions to compute a score per student.
    Returns list of dicts: [{"student_key": "...", "student_name":"...", "student_telegram_id":"...", "score":123, "details": {...}}]
    student_key chosen as student_telegram_id if present else student_name
    """
    if not db:
        raise RuntimeError("DB not configured")

    # 1) Aggregate test attempts: sum scores grouped by student_key
    pipeline_attempts = []
    if class_for:
        pipeline_attempts.append({"$match": {"student_class": class_for}})
    pipeline_attempts += [
        {"$match": {"status": {"$in": ["submitted","graded"]}, "score": {"$ne": None}}},
        {"$group": {
            "_id": {
                "student_telegram_id": {"$ifNull": ["$student_telegram_id", None]},
                "student_name": "$student_name",
            },
            "total_test_score": {"$sum": "$score"},
            "attempts_count": {"$sum": 1}
        }},
    ]
    attempts_cursor = db.attempts.aggregate(pipeline_attempts)
    score_map: Dict[str, Dict[str, Any]] = {}

    async for row in attempts_cursor:
        key = row["_id"]["student_telegram_id"] or row["_id"]["student_name"]
        student_name = row["_id"]["student_name"]
        score = row.get("total_test_score", 0) * WEIGHT_TEST_SCORE
        score_map[key] = {
            "student_key": key,
            "student_name": student_name,
            "student_telegram_id": row["_id"].get("student_telegram_id"),
            "score": score,
            "details": {"test_score": row.get("total_test_score", 0), "attempts": row.get("attempts_count", 0)}
        }

    # 2) Aggregate homework (graded) scores
    pipeline_hw = []
    if class_for:
        pipeline_hw.append({"$match": {"student_class": class_for}})
    pipeline_hw += [
        {"$match": {"graded": True, "grade": {"$ne": None}}},
        {"$group": {
            "_id": {"student_telegram_id": {"$ifNull": ["$student_telegram_id", None]}, "student_name": "$student_name"},
            "total_hw_grade": {"$sum": "$grade"},
            "hw_count": {"$sum": 1}
        }}
    ]
    hw_cursor = db.homework_submissions.aggregate(pipeline_hw)
    async for row in hw_cursor:
        key = row["_id"]["student_telegram_id"] or row["_id"]["student_name"]
        hw_score = row.get("total_hw_grade", 0) * WEIGHT_HOMEWORK
        if key in score_map:
            score_map[key]["score"] += hw_score
            score_map[key]["details"].update({"hw_score": row.get("total_hw_grade", 0), "hw_count": row.get("hw_count", 0)})
        else:
            score_map[key] = {
                "student_key": key,
                "student_name": row["_id"]["student_name"],
                "student_telegram_id": row["_id"].get("student_telegram_id"),
                "score": hw_score,
                "details": {"hw_score": row.get("total_hw_grade", 0), "hw_count": row.get("hw_count", 0)}
            }

    # 3) Doubt submissions count (reward engagement)
    pipeline_doubt = []
    if class_for:
        pipeline_doubt.append({"$match": {"student_class": class_for}})
    pipeline_doubt += [
        {"$group": {
            "_id": {"student_telegram_id": {"$ifNull": ["$student_telegram_id", None]}, "student_name": "$student_name"},
            "doubt_count": {"$sum": 1}
        }}
    ]
    doubt_cursor = db.doubts.aggregate(pipeline_doubt)
    async for row in doubt_cursor:
        key = row["_id"]["student_telegram_id"] or row["_id"]["student_name"]
        add = row.get("doubt_count", 0) * WEIGHT_DOUBT_SUBMIT
        if key in score_map:
            score_map[key]["score"] += add
            score_map[key]["details"].update({"doubts": row.get("doubt_count", 0)})
        else:
            score_map[key] = {
                "student_key": key,
                "student_name": row["_id"]["student_name"],
                "student_telegram_id": row["_id"].get("student_telegram_id"),
                "score": add,
                "details": {"doubts": row.get("doubt_count", 0)}
            }

    # 4) Activity bonus (simple: number of submissions total)
    # We can compute as attempts + hw + doubts count already in details; skip extra heavy ops.

    # convert to list and sort
    results = list(score_map.values())
    results.sort(key=lambda x: x["score"], reverse=True)
    # add rank
    for idx, r in enumerate(results, start=1):
        r["rank"] = idx
    return results

# Admin: compute & store snapshot (protected)
@app.post("/admin/leaderboard/compute", response_model=Dict[str, Any])
async def compute_leaderboard_snapshot(snapshot_name: Optional[str] = None, class_for: Optional[str] = None, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    """
    Compute leaderboard and store snapshot in db.leaderboard collection.
    snapshot_name optional (e.g., 'global_2025-11-16')
    """
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    snapshot_name = snapshot_name or f"global_{datetime.utcnow().date().isoformat()}"
    results = await _compute_scores_by_student(class_for=class_for)
    doc = {
        "snapshot_name": snapshot_name,
        "class_for": class_for,
        "computed_at": datetime.utcnow(),
        "entries": results,
        "total": len(results)
    }
    await db.leaderboard.replace_one({"snapshot_name": snapshot_name, "class_for": class_for}, doc, upsert=True)
    return {"ok": True, "snapshot": snapshot_name, "count": len(results)}

# Public: get latest snapshot (global or class)
@app.get("/leaderboard/global", response_model=Dict[str, Any])
async def get_global_leaderboard(limit: int = 10, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    doc = await db.leaderboard.find_one(sort=[("computed_at", DESCENDING)])
    if not doc:
        # compute on-the-fly if none exists
        results = await _compute_scores_by_student()
        return {"source": "computed", "total": len(results), "entries": results[:limit]}
    entries = doc.get("entries", [])[:limit]
    return {"source": "snapshot", "snapshot_name": doc.get("snapshot_name"), "computed_at": doc.get("computed_at"), "total": doc.get("total"), "entries": entries}

# Public: class-wise leaderboard (latest snapshot for that class or compute)
@app.get("/leaderboard/class/{class_for}", response_model=Dict[str, Any])
async def get_class_leaderboard(class_for: str, limit: int = 10, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    doc = await db.leaderboard.find_one({"class_for": class_for}, sort=[("computed_at", DESCENDING)])
    if not doc:
        results = await _compute_scores_by_student(class_for=class_for)
        return {"source": "computed", "class_for": class_for, "total": len(results), "entries": results[:limit]}
    return {"source": "snapshot", "class_for": class_for, "snapshot_name": doc.get("snapshot_name"), "computed_at": doc.get("computed_at"), "total": doc.get("total"), "entries": doc.get("entries", [])[:limit]}

# Public: get single student's rank (by telegram id or name)
@app.get("/leaderboard/student/{student_key}", response_model=Dict[str, Any])
async def get_student_rank(student_key: str, class_for: Optional[str] = None, db=Depends(get_db)):
    """
    student_key may be telegram id or student name.
    """
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    # try to find latest snapshot for class if present else global
    doc = None
    if class_for:
        doc = await db.leaderboard.find_one({"class_for": class_for}, sort=[("computed_at", DESCENDING)])
    if not doc:
        doc = await db.leaderboard.find_one(sort=[("computed_at", DESCENDING)])
    if doc:
        entries = doc.get("entries", [])
        for e in entries:
            if e.get("student_key") == student_key or (e.get("student_telegram_id") and str(e.get("student_telegram_id")) == student_key) or (e.get("student_name") and e.get("student_name").lower() == student_key.lower()):
                return {"found": True, "entry": e, "snapshot": doc.get("snapshot_name"), "computed_at": doc.get("computed_at")}
        # not found in snapshot -> compute on the fly
    results = await _compute_scores_by_student(class_for=class_for)
    for e in results:
        if e.get("student_key") == student_key or (e.get("student_telegram_id") and str(e.get("student_telegram_id")) == student_key) or (e.get("student_name") and e.get("student_name").lower() == student_key.lower()):
            return {"found": True, "entry": e, "source": "computed"}
    return {"found": False}

# Admin: list snapshots
@app.get("/admin/leaderboard/snapshots", response_model=List[Dict[str, Any]])
async def list_snapshots(limit: int = 20, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    cursor = db.leaderboard.find().sort("computed_at", DESCENDING).limit(min(limit, 100))
    out = []
    async for d in cursor:
        out.append({"snapshot_name": d.get("snapshot_name"), "class_for": d.get("class_for"), "computed_at": d.get("computed_at"), "total": d.get("total")})
    return out

# ---------------------------
# NOTIFICATION & REMINDER ENGINE
# ---------------------------

class AnnouncementIn(BaseModel):
    title: str
    message: str
    class_for: Optional[str] = None   # null → send to everyone


@app.post("/admin/announce", response_model=Dict[str, Any])
async def admin_announce(payload: AnnouncementIn, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    """
    Announcement broadcast: saves to db + sends Telegram messages (background)
    """
    if not db:
        raise HTTPException(503, "db-not-configured")

    doc = {
        "title": payload.title,
        "message": payload.message,
        "class_for": payload.class_for,
        "created_at": datetime.utcnow()
    }
    res = await db.notifications.insert_one(doc)

    # students to notify
    q = {}
    if payload.class_for:
        q["klass"] = payload.class_for
    cursor = db.students.find(q)

    async for s in cursor:
        tg = s.get("telegram_id")
        if tg:
            asyncio.create_task(
                send_telegram_message(TELEGRAM_BOT_TOKEN, tg, f"📢 {payload.title}\n\n{payload.message}")
            )

    return {"ok": True, "id": str(res.inserted_id)}


@app.get("/student/notifications/{student_id}", response_model=List[Dict[str, Any]])
async def get_student_notifications(student_id: str, db=Depends(get_db)):
    """
    Fetch notifications for a specific student (class-based + global)
    """
    if not db:
        raise HTTPException(503, "db-not-configured")

    try:
        sid = ObjectId(student_id)
    except:
        raise HTTPException(400, "invalid-id")

    student = await db.students.find_one({"_id": sid})
    if not student:
        raise HTTPException(404, "not-found")

    klass = student.get("klass")

    cursor = db.notifications.find({
        "$or": [
            {"class_for": None},          # global notifications
            {"class_for": klass}         # class-specific
        ]
    }).sort("created_at", -1)

    out = []
    async for n in cursor:
        out.append(serialize_doc(n))
    return out


# ---------------------------
# DAILY REMINDER SCHEDULER (EMULATED CRON)
# ---------------------------

@app.post("/admin/reminder/add", response_model=Dict[str, Any])
async def add_reminder(payload: Dict[str, str], db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    """
    payload: { "type": "chapter"|"homework"|"test"|"live", "time": "08:00", "class_for": "10" }
    """
    if not db:
        raise HTTPException(503, "db-not-configured")

    doc = {
        "type": payload.get("type"),
        "time": payload.get("time"),
        "class_for": payload.get("class_for"),
        "created_at": datetime.utcnow()
    }
    res = await db.reminders.insert_one(doc)
    return {"ok": True, "id": str(res.inserted_id)}


@app.post("/admin/reminder/run", response_model=Dict[str, Any])
async def run_reminders(db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    """
    Worker: run manually or every 30 mins via Render Cron
    Will check reminders & send messages.
    """
    if not db:
        raise HTTPException(503, "db-not-configured")

    now = datetime.utcnow().strftime("%H:%M")
    cursor = db.reminders.find({"time": now})

    triggered = 0

    async for r in cursor:
        klass = r.get("class_for")
        q = {"klass": klass} if klass else {}
        students = db.students.find(q)

        async for s in students:
            tg = s.get("telegram_id")
            if tg:
                text = f"🔔 Reminder: {r.get('type').title()} for class {klass or 'All'}"
                asyncio.create_task(send_telegram_message(TELEGRAM_BOT_TOKEN, tg, text))
        triggered += 1

    return {"ok": True, "triggered": triggered}

# ---------------------------
# STUDENT LEARNING FLOW + AI AUTO-TEACHER + Tutor Chat
# ---------------------------
from uuid import uuid4
from typing import Any

# Collections used:
# db.courses         -> optional course syllabus (can be populated or generated on demand)
# db.course_progress -> per-student progress through a course
# db.lessons_cache   -> cache for generated lessons to avoid repeat cost

# Pydantic models
class CourseStartIn(BaseModel):
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    subject: str
    klass: ShortClass
    mode: Optional[str] = "self-paced"  # auto / self-paced
    chapters: Optional[List[str]] = None  # optional list of chapter titles; if None AI will generate syllabus

class NextLessonOut(BaseModel):
    progress_id: str
    lesson_id: str
    lesson_title: str
    lesson_text: str
    notes: Optional[str] = None
    quiz: Optional[List[Dict[str, Any]]] = None
    audio_base64: Optional[str] = None

class CompleteIn(BaseModel):
    feedback: Optional[str] = None
    took_minutes: Optional[int] = None
    score: Optional[float] = None

class ChatIn(BaseModel):
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    subject: Optional[str] = None
    message: str
    context: Optional[List[Dict[str,str]]] = None  # previous messages if any

# Helper: basic syllabus generator (AI-assisted)
async def ensure_syllabus(subject: str, klass: str, chapters: Optional[List[str]] = None) -> List[str]:
    """
    If chapters provided, return them. Otherwise check db.courses for existing syllabus key,
    else call AI to generate a chapter list (10-20 items) and store.
    """
    key = f"{subject.lower()}_class_{klass}"
    existing = await db.courses.find_one({"key": key})
    if existing and existing.get("chapters"):
        return existing["chapters"]
    if chapters and len(chapters) > 0:
        doc = {"key": key, "subject": subject, "class": klass, "chapters": chapters, "created_at": datetime.utcnow()}
        await db.courses.insert_one(doc)
        return chapters

    # Ask AI to produce syllabus
    prompt = f"Generate a comprehensive list of 12-18 chapter titles for Class {klass} {subject} syllabus suitable for learning in sequence. Return as JSON array of titles."
    raw = await ai_generate(prompt)
    # try parsing JSON-ish output
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            chapters = parsed
        else:
            # fallback: split by newline
            chapters = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    except Exception:
        chapters = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]

    # trim and store
    chapters = [c for c in chapters if c][:18]
    await db.courses.insert_one({"key": key, "subject": subject, "class": klass, "chapters": chapters, "created_at": datetime.utcnow()})
    return chapters

# Helper: generate lesson content (AI) and cache
async def generate_lesson(subject: str, klass: str, chapter_title: str) -> Dict[str, Any]:
    cache_key = f"lesson::{subject}::{klass}::{chapter_title}"
    cached = await db.lessons_cache.find_one({"key": cache_key})
    if cached:
        return cached["value"]

    # Build AI prompt for a lesson
    prompt = f"""
    Create a lesson for Class {klass} {subject} on the chapter titled: "{chapter_title}".
    Provide:
    1) Short explanation in simple Hindi/English (use mix) in step-by-step points (max 600 words).
    2) 5 example problems with short solutions.
    3) 5 MCQ questions in JSON with options and answer index.
    4) A short summary (5 bullets).
    Output as JSON with keys: lesson_text, examples (list), quiz (list of objects), summary (list).
    """
    raw = await ai_generate(prompt)
    # Attempt to parse JSON; if fails, wrap raw in lesson_text
    lesson_obj = {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            lesson_obj = parsed
        else:
            lesson_obj = {"lesson_text": raw, "examples": [], "quiz": [], "summary": []}
    except Exception:
        lesson_obj = {"lesson_text": raw, "examples": [], "quiz": [], "summary": []}

    # Optionally generate audio (TTS) - best-effort
    audio_b64 = None
    try:
        tts_resp = await ai_generate(f"Convert this to a short spoken paragraph in Hindi/English suitable for students: {lesson_obj.get('lesson_text','')[:800]}")
        # if ai_generate returns base64 or raw audio bytes not supported here - skip heavy
        # We'll not attempt binary TTS here to avoid complexity; client can call /ai/voice with lesson_text.
    except Exception:
        pass

    value = {"lesson_text": lesson_obj.get("lesson_text"), "examples": lesson_obj.get("examples"), "quiz": lesson_obj.get("quiz"), "summary": lesson_obj.get("summary"), "cached_at": datetime.utcnow()}
    await db.lessons_cache.insert_one({"key": cache_key, "value": value, "created_at": datetime.utcnow()})
    return value

# Start course / create progress
@app.post("/student/learn/start", response_model=Dict[str, Any])
async def student_learn_start(payload: CourseStartIn, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    # ensure syllabus
    chapters = await ensure_syllabus(payload.subject, payload.klass, payload.chapters)
    # create progress doc
    progress_id = str(uuid4())
    doc = {
        "progress_id": progress_id,
        "student_id": payload.student_id,
        "student_name": payload.student_name,
        "subject": payload.subject,
        "klass": payload.klass,
        "chapters": chapters,
        "current_index": 0,
        "completed": [],
        "mode": payload.mode,
        "started_at": datetime.utcnow(),
        "last_update": datetime.utcnow()
    }
    await db.course_progress.insert_one(doc)
    return {"ok": True, "progress_id": progress_id, "chapters_count": len(chapters)}

# Get next lesson for a given progress
@app.get("/student/learn/next/{progress_id}", response_model=NextLessonOut)
async def student_learn_next(progress_id: str, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    prog = await db.course_progress.find_one({"progress_id": progress_id})
    if not prog:
        raise HTTPException(status_code=404, detail="progress-not-found")
    idx = int(prog.get("current_index", 0))
    chapters = prog.get("chapters", [])
    if idx >= len(chapters):
        return {"progress_id": progress_id, "lesson_id": None, "lesson_title": None, "lesson_text": "Course completed!", "notes": None, "quiz": None, "audio_base64": None}
    chapter_title = chapters[idx]
    lesson = await generate_lesson(prog["subject"], prog["klass"], chapter_title)
    lesson_id = f"{progress_id}::{idx}"
    # return lesson + quiz (if generated)
    return {
        "progress_id": progress_id,
        "lesson_id": lesson_id,
        "lesson_title": chapter_title,
        "lesson_text": lesson.get("lesson_text"),
        "notes": "\n".join(lesson.get("summary") or []),
        "quiz": lesson.get("quiz"),
        "audio_base64": None
    }

# Mark current lesson complete and advance
@app.post("/student/learn/complete/{progress_id}", response_model=Dict[str, Any])
async def student_learn_complete(progress_id: str, payload: CompleteIn, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    prog = await db.course_progress.find_one({"progress_id": progress_id})
    if not prog:
        raise HTTPException(status_code=404, detail="progress-not-found")
    idx = int(prog.get("current_index", 0))
    chapters = prog.get("chapters", [])
    if idx >= len(chapters):
        return {"ok": False, "message": "already completed"}
    # record completion
    entry = {"chapter": chapters[idx], "completed_at": datetime.utcnow(), "feedback": payload.feedback, "took_minutes": payload.took_minutes, "score": payload.score}
    await db.course_progress.update_one({"progress_id": progress_id}, {"$push": {"completed": entry}, "$inc": {"current_index": 1}, "$set": {"last_update": datetime.utcnow()}})
    new_prog = await db.course_progress.find_one({"progress_id": progress_id})
    return {"ok": True, "next_index": new_prog.get("current_index", 0), "completed_len": len(new_prog.get("completed", []))}

# Get progress status
@app.get("/student/learn/progress/{progress_id}", response_model=Dict[str, Any])
async def student_learn_progress(progress_id: str, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    prog = await db.course_progress.find_one({"progress_id": progress_id})
    if not prog:
        raise HTTPException(status_code=404, detail="progress-not-found")
    # compute completion %
    total = len(prog.get("chapters", []))
    done = len(prog.get("completed", []))
    percent = int((done / total) * 100) if total > 0 else 0
    return {"progress_id": progress_id, "student_name": prog.get("student_name"), "subject": prog.get("subject"), "klass": prog.get("klass"), "current_index": prog.get("current_index"), "completed": prog.get("completed"), "percent": percent}

# Auto-Course Mode: pushes next lesson automatically (client polls or webhook)
@app.post("/student/learn/auto_course_start", response_model=Dict[str, Any])
async def auto_course_start(payload: CourseStartIn, db=Depends(get_db), background_tasks: BackgroundTasks = None):
    # start progress
    res = await student_learn_start(payload, db=db)
    progress_id = res["progress_id"]
    # if mode=auto, optionally schedule next lessons notifications - here we just return token and client should poll.
    return {"ok": True, "progress_id": progress_id, "mode": payload.mode, "note": "Auto-course started. Client should poll /student/learn/next/<progress_id> or subscribe to notifications."}

# Conversational AI Tutor Chat (context aware)
@app.post("/ai/chat_tutor", response_model=Dict[str, Any])
async def ai_chat_tutor(payload: ChatIn):
    """
    Conversational tutor. Maintain context if provided (list of {"role":"user"/"assistant","content":"..."}).
    """
    # Build prompt with optional context
    messages = []
    if payload.context:
        # context should be list of role/content
        for m in payload.context:
            try:
                role = m.get("role", "user")
                content = m.get("content", "")
                messages.append({"role": role, "content": content})
            except:
                continue
    # append user question
    messages.append({"role": "user", "content": payload.message})
    # For ai_generate we only accept single prompt string; pack context
    prompt = "Conversation context:\n"
    for m in messages:
        prompt += f"{m['role'].upper()}: {m['content']}\n"
    prompt += "\nRespond as a helpful, patient teacher. Explain step by step and give examples."
    resp = await ai_generate(prompt)
    return {"reply": resp, "raw": resp}

# ---------------------------
# NEXT LESSON WITH SERVER-SIDE TTS (audio attach)
# ---------------------------
import base64

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")  # change if required

async def generate_tts_base64(text: str, voice: str = "alloy") -> Optional[str]:
    """
    Call OpenAI TTS endpoint and return base64 audio (mp3/ogg) or None on failure.
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY missing — TTS disabled")
        return None
    try:
        url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        body = {
            "model": TTS_MODEL,
            "voice": voice,
            "input": text
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                logger.warning("TTS failed status=%s text=%s", resp.status_code, resp.text)
                return None
            audio_bytes = resp.content
            return base64.b64encode(audio_bytes).decode()
    except Exception as e:
        logger.exception("TTS request failed: %s", e)
        return None

@app.get("/student/learn/next_with_audio/{progress_id}", response_model=Dict[str, Any])
async def student_learn_next_with_audio(progress_id: str, db=Depends(get_db)):
    """
    Same as /student/learn/next but also attempts to attach TTS audio (base64).
    Warning: TTS call may add latency; client can use /ai/voice separately if preferred.
    """
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    prog = await db.course_progress.find_one({"progress_id": progress_id})
    if not prog:
        raise HTTPException(status_code=404, detail="progress-not-found")
    idx = int(prog.get("current_index", 0))
    chapters = prog.get("chapters", [])
    if idx >= len(chapters):
        return {"progress_id": progress_id, "lesson_id": None, "lesson_title": None, "lesson_text": "Course completed!", "notes": None, "quiz": None, "audio_base64": None}
    chapter_title = chapters[idx]
    lesson = await generate_lesson(prog["subject"], prog["klass"], chapter_title)
    lesson_id = f"{progress_id}::{idx}"

    # prepare short TTS-friendly text (trim to safe length)
    tts_text = lesson.get("lesson_text") or ""
    if len(tts_text) > 2000:
        tts_text = tts_text[:1900] + " ... summary shortened for audio."

    audio_b64 = await generate_tts_base64(tts_text)
    return {
        "progress_id": progress_id,
        "lesson_id": lesson_id,
        "lesson_title": chapter_title,
        "lesson_text": lesson.get("lesson_text"),
        "notes": "\n".join(lesson.get("summary") or []),
        "quiz": lesson.get("quiz"),
        "audio_base64": audio_b64
    }

# ---------------------------
# TTS AUDIO CACHING (generate & store base64 in db.lessons_cache)
# ---------------------------

# Reuse generate_lesson(subject, klass, chapter_title) from earlier.
# This adds caching for audio: stores {"key": cache_key, "value": {..., "audio_b64": "..."}}

async def get_cached_lesson_audio(subject: str, klass: str, chapter_title: str) -> Optional[str]:
    """
    Return cached audio_base64 if present, else None.
    """
    if not db:
        return None
    cache_key = f"lesson::{subject}::{klass}::{chapter_title}"
    rec = await db.lessons_cache.find_one({"key": cache_key})
    if rec and isinstance(rec.get("value"), dict):
        audio = rec["value"].get("audio_b64")
        if audio:
            logger.info("TTS cache hit for %s", cache_key)
            return audio
    return None

async def save_lesson_audio_cache(subject: str, klass: str, chapter_title: str, lesson_value: dict, audio_b64: str):
    """
    Save lesson content (lesson_value) + audio_b64 into lessons_cache (upsert).
    """
    if not db:
        return
    cache_key = f"lesson::{subject}::{klass}::{chapter_title}"
    doc = {
        "key": cache_key,
        "value": {**lesson_value, "audio_b64": audio_b64},
        "cached_at": datetime.utcnow()
    }
    await db.lessons_cache.replace_one({"key": cache_key}, doc, upsert=True)
    logger.info("Saved TTS cache for %s", cache_key)

# Updated TTS generator that checks cache first, else generates & caches
async def generate_or_get_tts_for_lesson(subject: str, klass: str, chapter_title: str, tts_voice: str = "alloy") -> Optional[str]:
    """
    Returns base64 audio string for lesson (from cache or freshly generated + cached).
    """
    # check cache
    cached = await get_cached_lesson_audio(subject, klass, chapter_title)
    if cached:
        return cached

    # generate lesson text (or load cached lesson content)
    lesson_obj = await generate_lesson(subject, klass, chapter_title)
    text = lesson_obj.get("lesson_text") or ""
    if not text:
        logger.warning("No lesson text to TTS for %s - %s - %s", subject, klass, chapter_title)
        return None

    # trim for TTS safety
    if len(text) > 2000:
        tts_text = text[:1900] + " ... summary shortened for audio."
    else:
        tts_text = text

    # call OpenAI TTS (reuse generate_tts_base64 if present) or direct call here
    audio_b64 = None
    try:
        audio_b64 = await generate_tts_base64(tts_text, voice=tts_voice)
    except Exception as e:
        logger.exception("TTS generation failed: %s", e)
        audio_b64 = None

    if audio_b64:
        # save both lesson content and audio to cache
        await save_lesson_audio_cache(subject, klass, chapter_title, lesson_obj, audio_b64)
    return audio_b64

# Admin endpoint: generate TTS for single lesson and cache it (protected)
@app.post("/admin/tts/generate_for_lesson", response_model=Dict[str, Any])
async def admin_generate_tts_for_lesson(subject: str, klass: str, chapter_title: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    audio = await generate_or_get_tts_for_lesson(subject, klass, chapter_title)
    if not audio:
        return {"ok": False, "message": "TTS generation failed"}
    return {"ok": True, "cached": True, "audio_length_bytes": len(audio)}

# Admin endpoint: bulk generate cache for a syllabus (CAUTION: cost/time heavy)
@app.post("/admin/tts/generate_bulk", response_model=Dict[str, Any])
async def admin_generate_tts_bulk(subject: str, klass: ShortClass, start_index: int = 0, end_index: Optional[int] = None, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    """
    Use carefully. This will iterate syllabus (db.courses) and generate+cache audio for lessons.
    Optionally limit by start_index and end_index (0-based, inclusive).
    """
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    key = f"{subject.lower()}_class_{klass}"
    course = await db.courses.find_one({"key": key})
    if not course or not course.get("chapters"):
        raise HTTPException(status_code=404, detail="syllabus-not-found")
    chapters = course["chapters"]
    if end_index is None:
        end_index = len(chapters) - 1
    # clamp
    start_index = max(0, int(start_index))
    end_index = min(len(chapters) - 1, int(end_index))
    generated = []
    failed = []
    for idx in range(start_index, end_index + 1):
        ch = chapters[idx]
        try:
            audio = await generate_or_get_tts_for_lesson(subject, klass, ch)
            if audio:
                generated.append({"index": idx, "chapter": ch})
            else:
                failed.append({"index": idx, "chapter": ch})
            # small delay to avoid rate spikes
            await asyncio.sleep(1)
        except Exception as e:
            logger.exception("Bulk TTS failed for %s: %s", ch, e)
            failed.append({"index": idx, "chapter": ch, "error": str(e)})
    return {"ok": True, "generated_count": len(generated), "failed_count": len(failed), "generated": generated, "failed": failed}

# Update student next_with_audio endpoint to prefer cached audio (if present) - optional helper
# If you already have /student/learn/next_with_audio calling generate_tts_base64 directly,
# replace that call with generate_or_get_tts_for_lesson(subject, klass, chapter_title)
# (No new endpoint necessary; admin can pre-generate using above).

# ---------------------------
# REAL-TIME CHAT (WebSocket) — Student <-> Teacher
# ---------------------------
from fastapi import WebSocket, WebSocketDisconnect, Query, Form
from fastapi.websockets import WebSocketState
from collections import defaultdict
import json
from typing import Set

# In-memory connection manager (simple)
class ConnectionManager:
    def __init__(self):
        # room_id -> set of websockets
        self.active_rooms: Dict[str, Set[WebSocket]] = defaultdict(set)
        # websocket -> meta (user_id, name, room)
        self.meta: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, room_id: str, user_id: str, user_name: str):
        await websocket.accept()
        self.active_rooms[room_id].add(websocket)
        self.meta[websocket] = {"room": room_id, "user_id": user_id, "user_name": user_name}
        logger.info("WS connect: %s joined %s", user_name, room_id)

    def disconnect(self, websocket: WebSocket):
        meta = self.meta.get(websocket)
        if meta:
            room = meta.get("room")
            if room and websocket in self.active_rooms.get(room, set()):
                self.active_rooms[room].remove(websocket)
            self.meta.pop(websocket, None)
            logger.info("WS disconnect: %s left %s", meta.get("user_name"), room)

    async def broadcast(self, room_id: str, message: Dict[str, Any]):
        # send to all websockets in room
        conns = list(self.active_rooms.get(room_id, []))
        text = json.dumps(message)
        for ws in conns:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(text)
            except Exception:
                logger.exception("Failed to send WS message to %s", ws)

manager = ConnectionManager()

# Helper to persist chat message
async def save_chat_message(room_id: str, user_id: str, user_name: str, text: str, db_obj=None):
    if not db:
        return
    doc = {
        "room_id": room_id,
        "user_id": user_id,
        "user_name": user_name,
        "text": text,
        "created_at": datetime.utcnow()
    }
    try:
        await db.chats.insert_one(doc)
    except Exception:
        logger.exception("Failed to save chat message")

# WebSocket endpoint
# Client connects: ws://<host>/ws/chat/{room_id}?user_id=8304416413&user_name=Mantu
@app.websocket("/ws/chat/{room_id}")
async def websocket_chat(websocket: WebSocket, room_id: str, user_id: str = Query(...), user_name: str = Query(...)):
    """
    Real-time chat websocket.
    Query params required: user_id, user_name
    Example: /ws/chat/class_123?user_id=8304416413&user_name=Mantu
    """
    await manager.connect(websocket, room_id, user_id, user_name)
    # notify room that user joined (optional)
    join_msg = {"type": "system", "text": f"{user_name} joined the chat.", "user_name": "system", "created_at": datetime.utcnow().isoformat()}
    await manager.broadcast(room_id, join_msg)
    try:
        while True:
            data = await websocket.receive_text()
            # expect JSON payload: {"text":"..."}
            try:
                payload = json.loads(data)
                text = payload.get("text", "")
            except Exception:
                text = data
            # save message
            await save_chat_message(room_id, user_id, user_name, text, db)
            # broadcast to room
            out = {"type": "message", "user_id": user_id, "user_name": user_name, "text": text, "created_at": datetime.utcnow().isoformat()}
            await manager.broadcast(room_id, out)
            # if teacher not present in room and this is from student -> send telegram notify to teacher (best-effort)
            # heuristic: rooms like class_<id> -> teacher chat id must be known via DB mapping or TEACHER_CHAT_ID fallback
            teacher_present = any((m.get("user_id") == TEACHER_CHAT_ID) for m in manager.meta.values())
            if not teacher_present and room_id.startswith("class_"):
                # notify teacher via Telegram of unseen chat message
                try:
                    notify_text = f"💬 New message in {room_id} from {user_name}:\n\n{text[:300]}"
                    asyncio.create_task(send_telegram_message(TELEGRAM_BOT_TOKEN, TEACHER_CHAT_ID, notify_text))
                except Exception:
                    logger.exception("Failed to schedule telegram notify for chat")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # notify others
        leave_msg = {"type": "system", "text": f"{user_name} left the chat.", "user_name": "system", "created_at": datetime.utcnow().isoformat()}
        await manager.broadcast(room_id, leave_msg)
    except Exception:
        manager.disconnect(websocket)
        logger.exception("Websocket error")

# HTTP fallback: post a chat message (useful for clients without WS)
@app.post("/chat/send", response_model=Dict[str, Any])
async def http_send_chat(room_id: str = Form(...), user_id: str = Form(...), user_name: str = Form(...), text: str = Form(...), db=Depends(get_db)):
    # save
    await save_chat_message(room_id, user_id, user_name, text, db)
    # broadcast to WS clients in room
    out = {"type": "message", "user_id": user_id, "user_name": user_name, "text": text, "created_at": datetime.utcnow().isoformat()}
    await manager.broadcast(room_id, out)
    return {"ok": True}

# Get chat history for a room (paginated)
@app.get("/chat/history/{room_id}", response_model=Dict[str, Any])
async def chat_history(room_id: str, limit: int = 100, skip: int = 0, db=Depends(get_db), _auth: Optional[Any] = Depends(require_teacher_api_key)):
    """
    Returns last `limit` messages for room_id (requires teacher API key for privacy).
    You can change auth if you want students to read their own room history.
    """
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    cursor = db.chats.find({"room_id": room_id}).sort("created_at", -1).skip(int(skip)).limit(int(limit))
    out = []
    async for m in cursor:
        out.append(serialize_doc(m))
    # return reversed to chronological order
    out.reverse()
    return {"room_id": room_id, "messages": out}

# List active rooms & counts (admin)
@app.get("/admin/chat/active_rooms", response_model=Dict[str, Any])
async def admin_active_rooms(_auth=Depends(require_teacher_api_key)):
    rooms = {room: len(conns) for room, conns in manager.active_rooms.items()}
    return {"active_rooms": rooms}

# Admin: clear chat history for a room (protected)
@app.delete("/admin/chat/clear/{room_id}", response_model=Dict[str, Any])
async def admin_clear_chat(room_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    await db.chats.delete_many({"room_id": room_id})
    return {"ok": True, "room_id": room_id, "cleared": True}

# ---------------------------
# NOTES on scaling (Redis / PubSub) and persistence
# ---------------------------
# - Current manager stores connections in-memory; works for single instance.
# - For multi-instance (Render multiple replicas), use Redis pub/sub or a WebSocket gateway:
#   * Each instance publishes outgoing messages to Redis channel "chat:<room_id>".
#   * Each instance subscribes to channels for rooms that have local clients.
#   * When a message arrives from Redis, instance forwards to its local websockets.
# - Alternatively use Socket.IO + redis-adapter or third-party realtime services (Pusher, Ably).
# - Message storage in Mongo ensures history available even if WS clients miss messages.

# ---------------------------
# PAYMENT & SUBSCRIPTION MODULE (Razorpay)
# ---------------------------
import hmac
import hashlib
import base64
import os
from typing import Any

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
PAYMENT_CURRENCY = os.getenv("PAYMENT_CURRENCY", "INR")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    logger.warning("Razorpay keys not set — payment endpoints will fail until configured.")

# Helper: call Razorpay API (httpx)
async def razorpay_post(path: str, payload: dict) -> dict:
    url = f"https://api.razorpay.com/v1{path}"
    async with httpx.AsyncClient(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET), timeout=20) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# Create a one-time order (frontend will use razorpay checkout)
@app.post("/payment/create_order", response_model=Dict[str, Any])
async def create_order(amount_rupees: float, receipt: Optional[str] = None, notes: Optional[Dict[str, Any]] = None, db=Depends(get_db)):
    """
    amount_rupees: amount in rupees (eg. 199.0)
    This creates a Razorpay order (amount in paise).
    Returns order_id and other data for frontend checkout.
    """
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(503, detail="payment-gateway-not-configured")
    amt_paise = int(round(amount_rupees * 100))
    payload = {
        "amount": amt_paise,
        "currency": PAYMENT_CURRENCY,
        "receipt": receipt or f"rcpt_{int(datetime.utcnow().timestamp())}",
        "payment_capture": 1,
        "notes": notes or {}
    }
    try:
        data = await razorpay_post("/orders", payload)
    except httpx.HTTPStatusError as e:
        logger.exception("Razorpay order create failed: %s", e)
        raise HTTPException(status_code=502, detail="razorpay-order-failed")
    # optionally save order in DB
    if db:
        await db.payments.insert_one({
            "order_id": data.get("id"),
            "amount": amt_paise,
            "currency": PAYMENT_CURRENCY,
            "receipt": payload["receipt"],
            "notes": payload["notes"],
            "status": "created",
            "created_at": datetime.utcnow()
        })
    return {"order": data}

# Verify payment after checkout (signature verify)
@app.post("/payment/verify", response_model=Dict[str, Any])
async def verify_payment(payload: Dict[str, str], db=Depends(get_db)):
    """
    Payload expected:
    {
      "razorpay_payment_id":"pay_XXX",
      "razorpay_order_id":"order_XXX",
      "razorpay_signature":"signature"
    }
    """
    required = ("razorpay_payment_id", "razorpay_order_id", "razorpay_signature")
    if not all(k in payload for k in required):
        raise HTTPException(400, detail="missing-params")
    rp_id = payload["razorpay_payment_id"]
    ro_id = payload["razorpay_order_id"]
    signature = payload["razorpay_signature"]
    # signature = hmac_sha256(order_id + "|" + payment_id, secret)
    msg = f"{ro_id}|{rp_id}".encode()
    expected = hmac.new(RAZORPAY_KEY_SECRET.encode(), msg, hashlib.sha256).hexdigest()
    ok = (expected == signature)
    # update DB payment status
    if db:
        await db.payments.update_one({"order_id": ro_id}, {"$set": {"payment_id": rp_id, "signature": signature, "verified": ok, "verified_at": datetime.utcnow()}})
    if not ok:
        raise HTTPException(status_code=400, detail="signature-mismatch")
    # success: you can mark subscription/order active here
    return {"ok": True, "order_id": ro_id, "payment_id": rp_id}

# Create a subscription (server creates subscription tied to a plan on Razorpay dashboard)
@app.post("/subscription/create", response_model=Dict[str, Any])
async def create_subscription(plan_id: str, customer_notify: bool = True, start_at: Optional[int] = None, db=Depends(get_db)):
    """
    plan_id: Razorpay plan id (create plan on Razorpay dashboard or via API)
    start_at: epoch seconds to start, optional
    """
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(503, detail="payment-gateway-not-configured")
    payload = {"plan_id": plan_id, "total_count": 12, "customer_notify": 1 if customer_notify else 0}
    if start_at:
        payload["start_at"] = int(start_at)
    try:
        data = await razorpay_post("/subscriptions", payload)
    except httpx.HTTPStatusError as e:
        logger.exception("Razorpay subscription create failed: %s", e)
        raise HTTPException(status_code=502, detail="razorpay-subscription-failed")
    # save subscription record
    if db:
        await db.subscriptions.insert_one({
            "subscription_id": data.get("id"),
            "plan_id": plan_id,
            "status": data.get("status"),
            "created_at": datetime.utcnow(),
            "raw": data
        })
    return {"subscription": data}

# Webhook: Razorpay will POST events here (set webhook secret in Razorpay dashboard)
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

@app.post("/payment/webhook")
async def payment_webhook(request: Request, db=Depends(get_db)):
    """
    Verify signature: header 'X-Razorpay-Signature'
    body raw used for HMAC SHA256 with webhook secret
    """
    try:
        body_bytes = await request.body()
        signature = request.headers.get("x-razorpay-signature") or request.headers.get("X-Razorpay-Signature")
        if not RAZORPAY_WEBHOOK_SECRET:
            logger.warning("Webhook secret not set - skipping verification")
            verified = True
        else:
            expected = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), body_bytes, hashlib.sha256).hexdigest()
            verified = (expected == signature)
        if not verified:
            logger.warning("Webhook signature mismatch")
            raise HTTPException(status_code=400, detail="invalid-signature")
        event = await request.json()
        event_type = event.get("event")
        logger.info("Razorpay webhook event: %s", event_type)
        # handle some events
        if db:
            await db.payment_webhooks.insert_one({"event": event_type, "payload": event, "received_at": datetime.utcnow()})
        # example: payment.captured
        if event_type == "payment.captured":
            payload = event.get("payload", {})
            payment = payload.get("payment", {}).get("entity", {})
            order_id = payment.get("order_id")
            pid = payment.get("id")
            # update payment record
            if db and order_id:
                await db.payments.update_one({"order_id": order_id}, {"$set": {"payment_id": pid, "status": "captured", "captured_at": datetime.utcnow()}})
        # example: subscription.activated or payment.failed etc
        return {"ok": True}
    except Exception as e:
        logger.exception("Webhook processing failed: %s", e)
        raise HTTPException(status_code=500, detail="webhook-failed")

# Helper: get payment / subscription status
@app.get("/payment/status/{order_id}")
async def payment_status(order_id: str, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    row = await db.payments.find_one({"order_id": order_id})
    if not row:
        raise HTTPException(status_code=404, detail="not-found")
    return serialize_doc(row)

@app.get("/subscription/{subscription_id}")
async def subscription_status(subscription_id: str, db=Depends(get_db)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    row = await db.subscriptions.find_one({"subscription_id": subscription_id})
    if not row:
        # try fetch from Razorpay API for realtime status
        url = f"https://api.razorpay.com/v1/subscriptions/{subscription_id}"
        async with httpx.AsyncClient(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)) as client:
            r = await client.get(url)
            if r.status_code != 200:
                raise HTTPException(status_code=404, detail="subscription-not-found")
            data = r.json()
            return {"subscription": data}
    return serialize_doc(row)

# ---------------------------
# ADMIN PANEL BACKEND (user/teacher/content/roles/permissions + S3 presigned upload)
# ---------------------------
import uuid
import boto3
from botocore.exceptions import ClientError

# Env for S3 (optional)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET")

# init s3 client only if creds available
s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET:
    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    logger.info("S3 client initialized")
else:
    logger.info("S3 not configured; presigned upload endpoints will be disabled")

# ---------- Models ----------
class CreateTeacherIn(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    password: Optional[str] = None  # if you want to allow login (store hashed in DB)
    roles: Optional[List[str]] = ["teacher"]  # e.g., ["teacher","admin"]

class UserRoleUpdate(BaseModel):
    user_id: str
    roles: List[str]

class ContentIn(BaseModel):
    title: str
    description: Optional[str] = None
    subject: Optional[str] = None
    class_for: Optional[str] = None
    content_type: Optional[str] = "chapter"  # chapter/video/notes/test
    url: Optional[str] = None
    is_premium: Optional[bool] = False
    tags: Optional[List[str]] = []

# ---------- Helpers ----------
def hash_password(raw: str) -> str:
    # simple hashing (replace by bcrypt in prod)
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()

def current_time_iso():
    return datetime.utcnow().isoformat()

# ---------- Admin: create teacher/user ----------
@app.post("/admin/create_teacher", response_model=Dict[str, Any])
async def admin_create_teacher(payload: CreateTeacherIn, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    doc = payload.dict()
    # store hashed password if provided
    if doc.get("password"):
        doc["password_hash"] = hash_password(doc.pop("password"))
    doc.update({"created_at": datetime.utcnow(), "roles": doc.get("roles", ["teacher"]), "is_active": True})
    res = await db.users.insert_one(doc)
    await db.admin_logs.insert_one({"action": "create_teacher", "user_id": str(res.inserted_id), "by": "admin", "ts": datetime.utcnow(), "payload": {"name": payload.name, "email": payload.email}})
    return {"ok": True, "id": str(res.inserted_id)}

# ---------- Admin: list users ----------
@app.get("/admin/users", response_model=List[Dict[str, Any]])
async def admin_list_users(limit: int = 200, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    cursor = db.users.find({}).sort("created_at", -1).limit(min(limit, 1000))
    out = []
    async for u in cursor:
        u.pop("password_hash", None)
        out.append(serialize_doc(u))
    return out

# ---------- Admin: set user roles ----------
@app.post("/admin/user/set_roles", response_model=Dict[str, Any])
async def admin_set_roles(payload: UserRoleUpdate, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        uid = ObjectId(payload.user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    await db.users.update_one({"_id": uid}, {"$set": {"roles": payload.roles}})
    await db.admin_logs.insert_one({"action": "set_roles", "user_id": payload.user_id, "roles": payload.roles, "by": "admin", "ts": datetime.utcnow()})
    return {"ok": True, "user_id": payload.user_id, "roles": payload.roles}

# ---------- Admin: deactivate / activate user ----------
@app.post("/admin/user/toggle_active/{user_id}", response_model=Dict[str, Any])
async def admin_toggle_user(user_id: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        uid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    user = await db.users.find_one({"_id": uid})
    if not user:
        raise HTTPException(status_code=404, detail="not-found")
    new_state = not user.get("is_active", True)
    await db.users.update_one({"_id": uid}, {"$set": {"is_active": new_state}})
    await db.admin_logs.insert_one({"action": "toggle_active", "user_id": user_id, "new_state": new_state, "by": "admin", "ts": datetime.utcnow()})
    return {"ok": True, "user_id": user_id, "is_active": new_state}

# ---------- Content management (CRUD) ----------
@app.post("/admin/content/create", response_model=Dict[str, Any])
async def admin_create_content(payload: ContentIn, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow(), "published": False, "created_by": "admin"})
    res = await db.content.insert_one(doc)
    await db.admin_logs.insert_one({"action": "create_content", "content_id": str(res.inserted_id), "by": "admin", "ts": datetime.utcnow(), "payload": {"title": payload.title}})
    return {"ok": True, "id": str(res.inserted_id)}

@app.get("/admin/content", response_model=List[Dict[str, Any]])
async def admin_list_content(limit: int = 500, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    cursor = db.content.find({}).sort("created_at", -1).limit(min(limit, 1000))
    out = []
    async for c in cursor:
        out.append(serialize_doc(c))
    return out

@app.patch("/admin/content/{content_id}", response_model=Dict[str, Any])
async def admin_update_content(content_id: str, payload: ContentIn, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        cid = ObjectId(content_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    await db.content.update_one({"_id": cid}, {"$set": payload.dict()})
    await db.admin_logs.insert_one({"action": "update_content", "content_id": content_id, "by": "admin", "ts": datetime.utcnow()})
    doc = await db.content.find_one({"_id": cid})
    return serialize_doc(doc)

@app.post("/admin/content/publish/{content_id}", response_model=Dict[str, Any])
async def admin_publish_content(content_id: str, publish: bool = True, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        cid = ObjectId(content_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    await db.content.update_one({"_id": cid}, {"$set": {"published": publish}})
    await db.admin_logs.insert_one({"action": "publish_content", "content_id": content_id, "publish": publish, "by": "admin", "ts": datetime.utcnow()})
    doc = await db.content.find_one({"_id": cid})
    return serialize_doc(doc)

# ---------- S3 presigned upload (optional) ----------
@app.get("/admin/upload/presigned", response_model=Dict[str, Any])
async def admin_get_presigned_upload(filename: str, content_type: str = "application/octet-stream", expires_in: int = 3600, _auth=Depends(require_teacher_api_key)):
    """
    Returns presigned PUT URL for client to upload binary directly to S3.
    """
    if not s3_client:
        raise HTTPException(status_code=503, detail="s3-not-configured")
    key = f"uploads/{uuid.uuid4().hex}_{filename}"
    try:
        url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": S3_BUCKET, "Key": key, "ContentType": content_type},
            ExpiresIn=int(expires_in)
        )
        return {"ok": True, "url": url, "key": key, "bucket": S3_BUCKET}
    except ClientError as e:
        logger.exception("Presigned URL error: %s", e)
        raise HTTPException(status_code=500, detail="presigned-failed")

# ---------- Admin logs & audit ----------
@app.get("/admin/logs", response_model=List[Dict[str, Any]])
async def admin_get_logs(limit: int = 200, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    cursor = db.admin_logs.find({}).sort("ts", -1).limit(min(limit, 1000))
    out = []
    async for r in cursor:
        out.append(serialize_doc(r))
    return out

# ---------- System settings (feature flags) ----------
@app.post("/admin/settings/set", response_model=Dict[str, Any])
async def admin_set_setting(key: str, value: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    await db.settings.update_one({"key": key}, {"$set": {"value": value, "updated_at": datetime.utcnow()}}, upsert=True)
    await db.admin_logs.insert_one({"action": "set_setting", "key": key, "value": value, "by": "admin", "ts": datetime.utcnow()})
    return {"ok": True, "key": key, "value": value}

@app.get("/admin/settings/get", response_model=Dict[str, Any])
async def admin_get_setting(key: str, db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    row = await db.settings.find_one({"key": key})
    if not row:
        raise HTTPException(status_code=404, detail="not-found")
    return {"key": key, "value": row.get("value")}

# ---------- Quick utilities ----------
@app.get("/admin/health/full", response_model=Dict[str, Any])
async def admin_full_health(db=Depends(get_db), _auth=Depends(require_teacher_api_key)):
    """
    Returns health for DB, S3, AI, Payment, etc. Useful for admin dashboard.
    """
    status = {"db": False, "s3": False, "ai": False, "payment": False}
    try:
        if db:
            await db.command("ping")
            status["db"] = True
    except Exception:
        status["db"] = False
    if s3_client:
        try:
            # small head object check (bucket)
            s3_client.head_bucket(Bucket=S3_BUCKET)
            status["s3"] = True
        except Exception:
            status["s3"] = False
    if OPENAI_API_KEY:
        status["ai"] = True
    if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
        status["payment"] = True
    return {"ok": True, "status": status, "checked_at": datetime.utcnow().isoformat()}

# ---------------------------
# PARENTS NOTIFICATION SYSTEM
# ---------------------------

class ParentRegisterIn(BaseModel):
    student_id: str
    parent_name: str
    relation: str = "Parent"
    phone: Optional[str] = None
    telegram_id: Optional[str] = None   # parent ka Telegram chat ID
    want_updates: bool = True

@app.post("/parent/register", response_model=dict)
async def register_parent(payload: ParentRegisterIn, db=Depends(get_db)):
    """
    Parent ko student se link karta hai. 
    Agar parent ke telegram_id hai to updates direct jayenge.
    """
    if not db:
        raise HTTPException(503, "db-not-configured")

    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow()})
    res = await db.parents.insert_one(doc)
    return {"ok": True, "parent_id": str(res.inserted_id)}

# -------- Helper: Notify Parents --------

async def notify_parents(student_id: str, title: str, message: str):
    """
    Student ke saare linked parents ko notify karega.
    Telegram par message send karega (if telegram_id exists).
    """
    if not db:
        return

    parents = db.parents.find({"student_id": student_id, "want_updates": True})

    async for p in parents:
        tg = p.get("telegram_id")
        if tg:
            try:
                text = f"👨‍👩‍👦 PARENT UPDATE\n\n📝 {title}\n\n{message}"
                asyncio.create_task(send_telegram_message(TELEGRAM_BOT_TOKEN, tg, text))
            except Exception:
                logger.exception("Failed to notify parent")

# ----------- Auto Triggers -----------

# 1) Homework Submitted → Parent Notification
@app.post("/event/homework_submitted", response_model=dict)
async def event_homework_submitted(student_id: str, homework_title: str, score: Optional[int] = None, db=Depends(get_db)):
    msg = f"Your child has submitted the homework: **{homework_title}**."
    if score is not None:
        msg += f"\nScore: {score}/10"
    await notify_parents(student_id, "Homework Submitted", msg)
    return {"ok": True}

# 2) Test Result → Parent Update
@app.post("/event/test_result", response_model=dict)
async def event_test_result(student_id: str, test_title: str, score: float, total: float, db=Depends(get_db)):
    percent = round((score / total) * 100, 2)
    msg = f"Test: {test_title}\nScore: {score}/{total} ({percent}%)."
    await notify_parents(student_id, "Test Result", msg)
    return {"ok": True}

# 3) Attendance Update → Parent Alert
@app.post("/event/attendance", response_model=dict)
async def event_attendance(student_id: str, class_title: str, present: bool, db=Depends(get_db)):
    msg = f"Class: {class_title}\nStatus: {'Present' if present else 'Absent'}"
    await notify_parents(student_id, "Attendance Update", msg)
    return {"ok": True}

# 4) AI Lesson Completed → Parent Progress
@app.post("/event/lesson_completed", response_model=dict)
async def event_lesson_completed(student_id: str, chapter_title: str, time_spent: Optional[int] = None, db=Depends(get_db)):
    msg = f"Your child completed the chapter: {chapter_title}"
    if time_spent:
        msg += f"\nTime Spent: {time_spent} minutes."
    await notify_parents(student_id, "Lesson Completed", msg)
    return {"ok": True}

# 5) Doubt Asked → Parent Info (optional)
@app.post("/event/doubt_asked", response_model=dict)
async def event_doubt_asked(student_id: str, subject: str, db=Depends(get_db)):
    msg = f"Your child asked a doubt in **{subject}**. Great learning!"
    await notify_parents(student_id, "Doubt Asked", msg)
    return {"ok": True}

# 6) Subscription / Fee Payment Notification
@app.post("/event/payment_update", response_model=dict)
async def event_payment_update(student_id: str, amount: float, status: str, db=Depends(get_db)):
    msg = f"Payment Status: {status}\nAmount: ₹{amount}"
    await notify_parents(student_id, "Payment Update", msg)
    return {"ok": True}

# 7) Daily Summary (cron-based)
@app.post("/event/daily_summary", response_model=dict)
async def event_daily_summary(student_id: str, summary: str, db=Depends(get_db)):
    msg = f"Daily Summary:\n\n{summary}"
    await notify_parents(student_id, "Daily Summary", msg)
    return {"ok": True}

# ---------------------------
# PARENT DASHBOARD + WEEKLY PDF REPORT
# ---------------------------
import io
import base64
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from starlette.responses import StreamingResponse

# ---------- Parent dashboard endpoint ----------
@app.get("/parent/dashboard/{parent_id}", response_model=Dict[str, Any])
async def parent_dashboard(parent_id: str, db=Depends(get_db)):
    """
    Aggregated data for parent:
    - linked students
    - each student's progress %, recent tests, last homeworks, attendance summary
    """
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        pid = ObjectId(parent_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-id")
    parent = await db.parents.find_one({"_id": pid})
    if not parent:
        raise HTTPException(status_code=404, detail="parent-not-found")
    student_id = parent.get("student_id")
    # allow multiple students later; for now single student_id stored
    students = []
    if student_id:
        # try to read student doc
        try:
            st = await db.students.find_one({"_id": ObjectId(student_id)})
        except Exception:
            st = None
        if st:
            # compute progress%
            prog_doc = await db.course_progress.find_one({"student_id": student_id})
            if prog_doc:
                total = len(prog_doc.get("chapters", []))
                done = len(prog_doc.get("completed", []))
                percent = int((done / total) * 100) if total > 0 else 0
            else:
                percent = 0
            # recent tests (last 5 attempts)
            attempts_cursor = db.attempts.find({"student_telegram_id": st.get("telegram_id")}).sort("created_at", -1).limit(5)
            attempts = []
            async for a in attempts_cursor:
                attempts.append({"title": a.get("test_id"), "score": a.get("score"), "created_at": a.get("created_at")})
            # recent homeworks (last 5)
            hw_cursor = db.homework_submissions.find({"student_name": st.get("name")}).sort("created_at", -1).limit(5)
            homeworks = []
            async for h in hw_cursor:
                homeworks.append({"title": h.get("file_name") or h.get("homework_id"), "grade": h.get("grade"), "status": h.get("status"), "created_at": h.get("created_at")})
            # attendance summary (last 30 days)
            since = datetime.utcnow() - timedelta(days=30)
            att_count = await db.live_attendance.count_documents({"student_name": st.get("name"), "joined_at": {"$gte": since}})
            students.append({
                "student_id": student_id,
                "name": st.get("name"),
                "klass": st.get("klass"),
                "progress_percent": percent,
                "recent_tests": attempts,
                "recent_homeworks": homeworks,
                "attendance_last_30d": att_count
            })

    # notifications for parent (class/global)
    notifs = []
    cursor_n = db.notifications.find({"$or":[{"class_for": None}, {"class_for": st.get("klass") if students else None}] }).sort("created_at", -1).limit(10)
    async for n in cursor_n:
        notifs.append(serialize_doc(n))
    return {"parent_id": parent_id, "parent_name": parent.get("parent_name"), "students": students, "notifications": notifs}

# ---------- PDF report generator helper ----------
def build_weekly_pdf_bytes(parent_doc: dict, students_data: List[dict], week_start: datetime, week_end: datetime, generated_by: str = "Zynno") -> bytes:
    """
    Create a simple PDF bytes with summary for each linked student.
    """
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    # Header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(margin, y, f"Weekly Progress Report")
    p.setFont("Helvetica", 10)
    y -= 18
    p.drawString(margin, y, f"Parent: {parent_doc.get('parent_name')}    Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 14
    p.drawString(margin, y, f"Week: {week_start.date().isoformat()} to {week_end.date().isoformat()}")
    y -= 22

    for st in students_data:
        if y < 140:
            p.showPage()
            y = height - margin
        # Student header
        p.setFont("Helvetica-Bold", 12)
        p.drawString(margin, y, f"Student: {st.get('name')} (Class {st.get('klass')})")
        y -= 14
        p.setFont("Helvetica", 10)
        p.drawString(margin, y, f"Progress: {st.get('progress_percent')}%    Attendance (30d): {st.get('attendance_last_30d')}")
        y -= 14

        # Recent tests
        p.setFont("Helvetica-Bold", 11)
        p.drawString(margin, y, "Recent Tests:")
        y -= 12
        p.setFont("Helvetica", 10)
        if st.get("recent_tests"):
            for t in st["recent_tests"]:
                p.drawString(margin+10, y, f"- {t.get('title')} : {t.get('score')}")
                y -= 12
                if y < 80:
                    p.showPage(); y = height - margin
        else:
            p.drawString(margin+10, y, "No recent tests.")
            y -= 12

        # Recent homeworks
        p.setFont("Helvetica-Bold", 11)
        p.drawString(margin, y, "Recent Homeworks:")
        y -= 12
        p.setFont("Helvetica", 10)
        if st.get("recent_homeworks"):
            for h in st["recent_homeworks"]:
                p.drawString(margin+10, y, f"- {h.get('title')} : {h.get('grade') or 'N/A'} ({h.get('status')})")
                y -= 12
                if y < 80:
                    p.showPage(); y = height - margin
        else:
            p.drawString(margin+10, y, "No recent homeworks.")
            y -= 12

        p.line(margin, y, width - margin, y)
        y -= 14

    p.setFont("Helvetica-Oblique", 8)
    p.drawString(margin, 30, f"Report generated by {generated_by} — Zynno Education Engine")
    p.save()
    buffer.seek(0)
    return buffer.read()

# ---------- Parent weekly report endpoint (returns PDF stream) ----------
@app.get("/parent/report/weekly/{parent_id}")
async def parent_weekly_report(parent_id: str, student_id: Optional[str] = None, db=Depends(get_db)):
    """
    Generate weekly PDF report for given parent (and optional specific student).
    Returns StreamingResponse application/pdf so client can download.
    """
    if not db:
        raise HTTPException(status_code=503, detail="db-not-configured")
    try:
        pid = ObjectId(parent_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid-parent-id")
    parent = await db.parents.find_one({"_id": pid})
    if not parent:
        raise HTTPException(status_code=404, detail="parent-not-found")

    # week range: last 7 days
    week_end = datetime.utcnow()
    week_start = week_end - timedelta(days=7)

    students_data = []
    # if student_id param given use that, else use linked student_id
    ids = []
    if student_id:
        ids = [student_id]
    else:
        if parent.get("student_id"):
            ids = [parent.get("student_id")]

    for sid in ids:
        try:
            st = await db.students.find_one({"_id": ObjectId(sid)})
        except Exception:
            st = None
        if not st:
            continue
        # compute progress % from course_progress
        prog = await db.course_progress.find_one({"student_id": sid})
        total = len(prog.get("chapters", [])) if prog else 0
        done = len(prog.get("completed", [])) if prog else 0
        percent = int((done / total) * 100) if total > 0 else 0
        # recent attempts within week
        attempts = []
        cursor_a = db.attempts.find({"student_telegram_id": st.get("telegram_id"), "created_at": {"$gte": week_start}}).sort("created_at", -1)
        async for a in cursor_a:
            attempts.append({"title": a.get("test_id"), "score": a.get("score"), "date": a.get("created_at")})
        # homeworks submitted in week
        hws = []
        cursor_h = db.homework_submissions.find({"student_name": st.get("name"), "created_at": {"$gte": week_start}}).sort("created_at", -1)
        async for h in cursor_h:
            hws.append({"title": h.get("file_name") or h.get("homework_id"), "grade": h.get("grade"), "date": h.get("created_at")})
        # attendance entries in week
        att_count = await db.live_attendance.count_documents({"student_name": st.get("name"), "joined_at": {"$gte": week_start}})
        students_data.append({
            "student_id": sid,
            "name": st.get("name"),
            "klass": st.get("klass"),
            "progress_percent": percent,
            "recent_tests": attempts,
            "recent_homeworks": hws,
            "attendance_week": att_count
        })

    # build pdf bytes
    pdf_bytes = build_weekly_pdf_bytes(parent, students_data, week_start, week_end)
    # optional: save into GridFS for recordkeeping
    # if gridfs_bucket:
    #     await gridfs_bucket.upload_from_stream(f"parent_report_{parent_id}_{int(datetime.utcnow().timestamp())}.pdf", io.BytesIO(pdf_bytes))

    # return as file stream
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=weekly_report_parent_{parent_id}.pdf"
    })

# ---------- run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)

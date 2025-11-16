# main.py
# Full FastAPI backend with MongoDB (motor) and Telegram notify
import os
import asyncio
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import httpx
from motor.motor_asyncio import AsyncIOMotorClient

# load .env if present (local dev)
load_dotenv()

# --- Config from env ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # from BotFather
TEACHER_CHAT_ID = os.getenv("TEACHER_CHAT_ID")        # numeric string
MONGO_URI = os.getenv("MONGO_URI")                    # mongodb+srv://.../...

if not MONGO_URI:
    # app can still start (but DB endpoints will error); you can also raise here.
    print("WARNING: MONGO_URI not set. DB operations will fail until set.")

# --- App init ---
app = FastAPI(title="Zynno - Backend")

# DB client will be created on startup
mongo_client: Optional[AsyncIOMotorClient] = None
db = None  # type: ignore

# --- Pydantic payloads ---
class NotifyPayload(BaseModel):
    student_name: str
    student_class: str
    doubt_text: str
    save_to_db: Optional[bool] = True  # whether to save this doubt into DB


class DoubtDoc(BaseModel):
    student_name: str
    student_class: str
    doubt_text: str
    status: str = "new"  # new / answered
    created_at: Optional[float] = None


# --- Telegram send helper (async) ---
async def send_telegram_message_async(text: str):
    """Send message to teacher via Telegram Bot API (async)."""
    if not TELEGRAM_BOT_TOKEN or not TEACHER_CHAT_ID:
        print("Telegram token/ID not configured. Skipping Telegram send.")
        return {"ok": False, "reason": "telegram_not_configured"}

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TEACHER_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        # "disable_web_page_preview": True
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, data=payload)
            # print status for logs
            print("Telegram status:", resp.status_code, resp.text)
            return {"ok": resp.status_code == 200, "status_code": resp.status_code, "text": resp.text}
    except Exception as e:
        print("Telegram send error:", str(e))
        return {"ok": False, "error": str(e)}


# --- DB helpers ---
async def save_doubt_to_db(doc: dict):
    if not db:
        raise RuntimeError("DB not connected")
    col = db["doubts"]
    res = await col.insert_one(doc)
    return str(res.inserted_id)


# --- FastAPI Lifespan (startup/shutdown) ---
@app.on_event("startup")
async def startup_event():
    global mongo_client, db
    if MONGO_URI:
        mongo_client = AsyncIOMotorClient(MONGO_URI)
        # default DB name from connection string may be none; use 'zynno' or database from URI
        # If using srv without db, choose one:
        try:
            # if URI contains a default db name after '/', try to use it, else 'zynno'
            dbname = MONGO_URI.rsplit("/", 1)[-1].split("?")[0] or "zynno"
        except Exception:
            dbname = "zynno"
        db = mongo_client[dbname]
        print("MongoDB connected, using DB:", dbname)
    else:
        print("MONGO_URI not provided at startup. DB disabled.")


@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client
    if mongo_client:
        mongo_client.close()
        print("MongoDB connection closed.")


# --- Endpoints ---
@app.get("/")
async def root():
    return {"status": "ok", "service": "Zynno backend"}


@app.post("/notify_teacher")
async def notify_teacher(payload: NotifyPayload, background_tasks: BackgroundTasks):
    """
    Notify teacher on Telegram and save doubt to MongoDB (optional).
    Request JSON:
    {
      "student_name": "Mantu",
      "student_class": "12A",
      "doubt_text": "Question text",
      "save_to_db": true
    }
    """
    # build message
    text = (
        "<b>📚 New Doubt</b>\n\n"
        f"<b>Student:</b> {payload.student_name}\n"
        f"<b>Class:</b> {payload.student_class}\n"
        f"<b>Doubt:</b> {payload.doubt_text}\n\n"
        "Reply to help the student."
    )

    # schedule telegram send in background
    # use asyncio.create_task to run async function in background
    background_tasks.add_task(asyncio.create_task, send_telegram_message_async(text))

    # optionally save to DB
    saved_id = None
    if payload.save_to_db:
        doc = {
            "student_name": payload.student_name,
            "student_class": payload.student_class,
            "doubt_text": payload.doubt_text,
            "status": "new",
            "created_at": asyncio.get_event_loop().time()
        }
        try:
            saved_id = await save_doubt_to_db(doc)
        except Exception as e:
            print("DB save error:", str(e))
            # do not fail the endpoint if DB error, just inform
            return {"status": "queued", "telegram": "queued", "db": "failed", "db_error": str(e)}

    return {"status": "queued", "telegram": "queued", "db_id": saved_id}


@app.get("/doubts")
async def list_doubts(limit: int = 50):
    """List recent doubts (from Mongo)."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")
    col = db["doubts"]
    cursor = col.find().sort("created_at", -1).limit(limit)
    items = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        items.append(doc)
    return {"count": len(items), "doubts": items}


@app.post("/answer_doubt/{doubt_id}")
async def answer_doubt(doubt_id: str, answer_text: str):
    """Mark doubt answered and optionally message student (not implemented: student chat id)."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not connected")
    col = db["doubts"]
    res = await col.update_one({"_id": None}, {"$set": {"status": "answered"}})  # placeholder
    # NOTE: If you want to message a student directly, you need student's chat id stored.
    return {"ok": True, "note": "Implement student notify by saving student chat id when collecting doubt."}

# add at top if not present
import os
import requests
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # from Render env
TEACHER_CHAT_ID = os.getenv("TEACHER_CHAT_ID")   # your ID (8304416413)

app = FastAPI(title="ZYNNO - Starter Backend (with Telegram)")

class NotifyPayload(BaseModel):
    student_name: str
    student_class: str
    doubt_text: str

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TEACHER_CHAT_ID:
        print("Telegram token/ID not configured.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TEACHER_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, data=data, timeout=10)
        print("Telegram response:", resp.status_code, resp.text)
    except Exception as e:
        print("Telegram send error:", e)

@app.post("/notify_teacher")
async def notify_teacher(payload: NotifyPayload, background_tasks: BackgroundTasks):
    text = (
        f"📚 <b>New Doubt</b>\n\n"
        f"<b>Student:</b> {payload.student_name}\n"
        f"<b>Class:</b> {payload.student_class}\n"
        f"<b>Doubt:</b> {payload.doubt_text}\n\n"
        f"Reply to help the student."
    )
    background_tasks.add_task(send_telegram_message, text)
    return {"status":"queued","message":"Teacher will be notified via Telegram."}

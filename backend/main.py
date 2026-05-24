from datetime import datetime, timedelta
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Header,
    HTTPException,
    status
)

from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError
from pydantic import BaseModel
from pypdf import PdfReader

from backend.rag import (
    retrieve_answer,
    add_document
)

from backend.memory import (
    save_memory,
    get_memory
)

# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AI SaaS RAG Backend",
    version="4.0"
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# SECURITY
# =========================================================

SECRET_KEY = "super_secret_key"

ALGORITHM = "HS256"

TOKEN_EXPIRE_MINUTES = 60

# =========================================================
# USERS
# =========================================================

users_db = {
    "rajesh": "1234",
    "admin": "admin123"
}

# =========================================================
# REQUEST MODELS
# =========================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    user: str
    message: str

# =========================================================
# TOKEN
# =========================================================

def create_token(username: str):

    expire = datetime.utcnow() + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": username,
        "exp": expire
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def verify_token(token: str):

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        return payload.get("sub")

    except JWTError:
        return None

# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "AI Backend Live"
    }

# =========================================================
# LOGIN
# =========================================================

@app.post("/login")
def login(data: LoginRequest):

    username = data.username
    password = data.password

    if username not in users_db:

        raise HTTPException(
            status_code=401,
            detail="Invalid username"
        )

    if users_db[username] != password:

        raise HTTPException(
            status_code=401,
            detail="Invalid password"
        )

    token = create_token(username)

    return {
        "message": "Login successful",
        "token": token,
        "user": username
    }

# =========================================================
# AUTH HELPER
# =========================================================

def authenticate_user(
    authorization: Optional[str]
):

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Authorization missing"
        )

    if not authorization.startswith("Bearer "):

        raise HTTPException(
            status_code=401,
            detail="Invalid token format"
        )

    token = authorization.replace(
        "Bearer ",
        ""
    )

    username = verify_token(token)

    if not username:

        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    return username

# =========================================================
# PDF UPLOAD
# =========================================================

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):

    username = authenticate_user(
        authorization
    )

    if file.content_type != "application/pdf":

        raise HTTPException(
            status_code=400,
            detail="Only PDF files allowed"
        )

    try:

        reader = PdfReader(file.file)

        text = ""

        for page in reader.pages:

            text += page.extract_text() or ""

        add_document(
            username,
            text
        )

        return {
            "message": "PDF uploaded successfully",
            "filename": file.filename,
            "characters": len(text)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================================================
# CHAT
# =========================================================

@app.post("/chat")
def chat(data: ChatRequest):

    try:

        result = retrieve_answer(
            data.user,
            data.message
        )

        save_memory(
            data.user,
            data.message,
            result["answer"]
        )

        history = get_memory(
            data.user
        )

        return {
            "response": result["answer"],
            "sources": result["sources"],
            "memory": len(history)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
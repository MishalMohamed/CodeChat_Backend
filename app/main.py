from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session

import app.models

from app.dependencies.auth_dependency import get_db, get_current_user

from app.models.user import User
from app.models.repository import Repository, RepoStatus

from app.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token
)

from app.services.repository_service import (
    create_repository,
    update_repository_status,
    get_repository_if_indexed
)
from app.services.chat_service import (
    save_chat,
    get_user_chats,
    get_repository_chats
)

from app.rag.rag_pipeline import RAGPipeline, RAGConfig
from app.services.chat_service import save_chat
from app.services.repository_service import get_repository_if_indexed

app = FastAPI()

rag_pipeline = RAGPipeline(RAGConfig())

# -----------------------------
# User Registration
# -----------------------------
@app.post("/register")
def register(username: str, password: str, db: Session = Depends(get_db)):

    existing_user = db.query(User).filter(User.username == username).first()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    hashed_pw = hash_password(password)

    new_user = User(
        username=username,
        password_hash=hashed_pw
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully"}


# -----------------------------
# User Login
# -----------------------------
@app.post("/login")
def login(username: str, password: str, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    access_token = create_access_token({"user_id": user.id})

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


# -----------------------------
# Add Repository
# -----------------------------
@app.post("/repository/add")
def add_repository(
    repo_url: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):

    repo = create_repository(
        db=db,
        user_id=current_user.id,
        repo_url=repo_url
    )

    return {
        "repository_id": repo.id,
        "repo_url": repo.repo_url,
        "status": repo.status
    }


# -----------------------------
# List User Repositories
# -----------------------------
@app.get("/repository/list")
def list_repositories(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):

    repos = db.query(Repository).filter(
        Repository.user_id == current_user.id
    ).all()

    return repos


# -----------------------------
# Check Repository Status
# -----------------------------
@app.get("/repository/status/{repo_id}")
def get_repository_status(
    repo_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):

    repo = db.query(Repository).filter(
        Repository.id == repo_id,
        Repository.user_id == current_user.id
    ).first()

    if not repo:
        raise HTTPException(
            status_code=404,
            detail="Repository not found"
        )

    return {
        "repo_id": repo.id,
        "status": repo.status,
        "faiss_index_path": repo.faiss_index_path
    }


# -----------------------------
# Update Repository Status
# (Used by indexing pipeline)
# -----------------------------
@app.post("/repository/update-status")
def update_repo_status(
    repo_id: int,
    status: RepoStatus,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):

    repo = update_repository_status(db, repo_id, status)

    if not repo:
        raise HTTPException(
            status_code=404,
            detail="Repository not found"
        )

    return {"message": "Repository status updated"}
@app.post("/chat/save")
def store_chat(
    repository_url: str,
    query_text: str,
    response_text: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):

    chat = save_chat(
        db=db,
        user_id=current_user.id,
        repository_url=repository_url,
        query_text=query_text,
        response_text=response_text
    )

    return {"message": "Chat saved", "chat_id": chat.id}
@app.get("/chat/history")
def get_chat_history(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):

    chats = get_user_chats(db, current_user.id)

    return chats
@app.get("/chat/repository")
def get_repo_chat_history(
    repository_url: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):

    chats = get_repository_chats(
        db,
        current_user.id,
        repository_url
    )

    return chats
from app.services.index_registry_service import get_index_path

@app.get("/repository/index-path/{repo_id}")
def fetch_index_path(
    repo_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):

    path = get_index_path(db, repo_id)

    return {"index_path": path}

@app.post("/chat/query")
def query_repository(
    repo_id: int,
    query: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    
    # Check repository belongs to user and is indexed
    repo = get_repository_if_indexed(db, repo_id)

    # Load FAISS index
    rag_pipeline.load_index(repo.faiss_index_path)

    # Run RAG query
    response = rag_pipeline.query(query)

    # Save chat history
    save_chat(
        db=db,
        user_id=current_user.id,
        repository_url=repo.repo_url,
        query_text=query,
        response_text=response.answer
    )

    return {
        "answer": response.answer,
        "sources": [
            {
                "file": meta.file_path,
                "symbol": meta.symbol_name,
                "line": meta.start_line
            }
            for meta, score in response.retrieved_chunks
        ]
    }
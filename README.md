CodeChat Backend

Implemented Features:

- User registration and login
- JWT authentication
- Password hashing using Passlib
- PostgreSQL database integration using SQLAlchemy
- Repository management and status tracking
- Chat history persistence
- Repository → FAISS index path mapping
- Protected API routes using authentication dependencies

Tech Stack:

- FastAPI  
- PostgreSQL  
- SQLAlchemy  
- JWT (python-jose)  
- Passlib (bcrypt)

## Run the Backend

Install dependencies:
```
pip install -r requirements.txt
```

Create database tables:
```
python create_tables.py
```

Run the server:
```
uvicorn app.main:app --reload
```

API documentation:
```
http://127.0.0.1:8000/docs
```

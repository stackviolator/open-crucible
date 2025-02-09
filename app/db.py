# app/db.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./users.db"  # or your preferred database URL

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}  # needed for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

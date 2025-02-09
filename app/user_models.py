# app/user_models.py

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from app.db import Base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=True)  # native login password hash
    highest_level = Column(Integer, default=1)
    oauth_provider = Column(String, nullable=True)  # e.g. "github"
    oauth_id = Column(String, nullable=True)        # GitHub user ID

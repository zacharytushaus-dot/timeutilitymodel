# storage.py — Postgres-ready models with safe connection handling for Streamlit
import os, uuid
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from sqlalchemy import create_engine, Column, String, JSON, ForeignKey, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import NullPool  # <--- KEY FIX

# --- DB URL resolver: env > st.secrets > local SQLite ---
def _db_url() -> str:
    # 1) Environment variable (works locally and in most hosts)
    url = os.getenv("DATABASE_URL")
    if url:
        return _normalize_db_url(url)

    # 2) Streamlit secrets when running via Streamlit with secrets.toml
    try:
        import streamlit as st  # only if available
        if st and "DATABASE_URL" in st.secrets:
            return _normalize_db_url(st.secrets["DATABASE_URL"])
    except Exception:
        pass

    # 3) Safe local fallback
    return "sqlite:///tum.db"


def _normalize_db_url(url: str) -> str:
    """
    Make sure we use SQLAlchemy's psycopg driver and add sane libpq keepalive params
    unless the caller already supplied their own query params.
    """
    # Upgrade old scheme to the modern SQLAlchemy dialect
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    # If caller already used postgresql+psycopg, leave as-is.

    # If it's not Postgres, return as-is (e.g., SQLite)
    if not url.startswith("postgresql+psycopg://"):
        return url

    # Append keepalive/ssl defaults only if user didn’t pass any params
    u = urlparse(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))

    # Only set defaults if the keys aren’t already present
    q.setdefault("sslmode", "require")
    q.setdefault("keepalives", "1")
    q.setdefault("keepalives_idle", "30")
    q.setdefault("keepalives_interval", "10")
    q.setdefault("keepalives_count", "5")

    u = u._replace(query=urlencode(q))
    return urlunparse(u)


DATABASE_URL = _db_url()

# Engine: FIXED to use NullPool
# We removed 'pool_size' and 'max_overflow'.
# poolclass=NullPool means "Don't hold connections open. Close them immediately after use."
engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    pool_recycle=300,
    poolclass=NullPool,  # <--- THIS PREVENTS THE ERROR
)

# Session: don't expire on commit; avoid autoflush surprises
Session = sessionmaker(
    bind=engine,
    future=True,
    expire_on_commit=False,
    autoflush=False,
)

Base = declarative_base()


# --- Models ---
class Org(Base):
    __tablename__ = "orgs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)  # will be the Supabase user id later
    email = Column(String, unique=True, nullable=False)
    org_id = Column(String, ForeignKey("orgs.id"), nullable=False)


class Client(Base):
    __tablename__ = "clients"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(String, ForeignKey("orgs.id"), nullable=True)  # stays null until auth step
    name = Column(String, nullable=False)
    email = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=text("now()"))  # works on PG and SQLite


class Run(Base):
    __tablename__ = "runs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(String, ForeignKey("orgs.id"), nullable=True)        # fill later
    operator_id = Column(String, ForeignKey("users.id"), nullable=True)  # fill later
    client_id = Column(String, ForeignKey("clients.id"), nullable=False)
    created_at = Column(DateTime, server_default=text("now()"))
    inputs = Column(JSON, nullable=False)   # becomes JSONB on Postgres
    outputs = Column(JSON, nullable=False)
    client = relationship("Client")


def ensure_tables():
    Base.metadata.create_all(engine)
    
"""SQLAlchemy models for vision data storage.

Tables:
    Photo: Indexed images with metadata (path, hash, dimensions, EXIF)
    Face: Detected faces with bounding boxes and embedding references
    Person: Named individuals for face identification
    Video: Indexed videos with duration and resolution
    VideoFrame: Extracted frames from videos with timestamps
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Generator

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from src.vision.config import SQLITE_PATH

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Photo(Base):
    """Indexed photo/image with metadata and analysis results.

    Attributes:
        id: Unique identifier (UUID or hash-based).
        path: Absolute path to the image file.
        hash: SHA256 hash for deduplication.
        width: Image width in pixels.
        height: Image height in pixels.
        taken_at: Date/time photo was taken (from EXIF).
        location_lat: GPS latitude (decimal degrees).
        location_lon: GPS longitude (decimal degrees).
        camera: Camera model from EXIF.
        description: VL-generated description of image content.
        indexed_at: When this photo was added to the database.
        faces: List of Face records detected in this photo.
    """

    __tablename__ = "photos"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    path = Column(Text, unique=True, nullable=False)
    hash = Column(String(64), nullable=False)  # SHA256 for dedup
    width = Column(Integer)
    height = Column(Integer)
    taken_at = Column(DateTime)
    location_lat = Column(Float)
    location_lon = Column(Float)
    camera = Column(String(128))
    description = Column(Text)
    indexed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    faces = relationship("Face", back_populates="photo", cascade="all, delete-orphan")


class Face(Base):
    """Detected face in a photo with bounding box and embedding reference.

    Attributes:
        id: Unique face identifier (UUID).
        photo_id: Foreign key to the parent Photo.
        person_id: Foreign key to identified Person (nullable).
        bbox_x: Bounding box left x-coordinate in pixels.
        bbox_y: Bounding box top y-coordinate in pixels.
        bbox_w: Bounding box width in pixels.
        bbox_h: Bounding box height in pixels.
        confidence: Detection confidence score (0-1).
        embedding_id: Reference to ArcFace embedding in ChromaDB.
        photo: Parent Photo relationship.
        person: Identified Person relationship (may be None).
    """

    __tablename__ = "faces"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    photo_id = Column(String(36), ForeignKey("photos.id", ondelete="CASCADE"), nullable=False)
    person_id = Column(String(36), ForeignKey("persons.id", ondelete="SET NULL"))
    bbox_x = Column(Integer, nullable=False)
    bbox_y = Column(Integer, nullable=False)
    bbox_w = Column(Integer, nullable=False)
    bbox_h = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    embedding_id = Column(String(36))  # Reference to ChromaDB

    # Relationships
    photo = relationship("Photo", back_populates="faces")
    person = relationship("Person", back_populates="faces")


class Person(Base):
    """Named individual for face identification and grouping.

    Persons are created either through face clustering (auto-generated ID)
    or manual identification (user-assigned name).

    Attributes:
        id: Unique person identifier (UUID).
        name: Display name (user-assigned, may be None for unidentified).
        photo_count: Cached count of photos containing this person.
        created_at: When this person record was created.
        faces: List of Face records identified as this person.
    """

    __tablename__ = "persons"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(256))
    photo_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    faces = relationship("Face", back_populates="person")


class Video(Base):
    """Indexed video with metadata and extracted frames.

    Videos are processed by extracting frames at a specified FPS rate.
    Each frame can be analyzed independently with the vision pipeline.

    Attributes:
        id: Unique video identifier (UUID).
        path: Absolute path to the video file.
        duration_secs: Video duration in seconds.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Original video frame rate.
        indexed_at: When this video was added to the database.
        frames: List of extracted VideoFrame records.
    """

    __tablename__ = "videos"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    path = Column(Text, unique=True, nullable=False)
    duration_secs = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    fps = Column(Float)
    indexed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    frames = relationship("VideoFrame", back_populates="video", cascade="all, delete-orphan")


class VideoFrame(Base):
    """Extracted frame from a video with analysis results.

    Frames are extracted at regular intervals and analyzed
    to enable video search and content understanding.

    Attributes:
        id: Unique frame identifier (UUID).
        video_id: Foreign key to the parent Video.
        timestamp_ms: Frame position in milliseconds from video start.
        thumbnail_path: Path to saved frame thumbnail image.
        description: VL-generated description of frame content.
        video: Parent Video relationship.
    """

    __tablename__ = "video_frames"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    timestamp_ms = Column(Integer, nullable=False)
    thumbnail_path = Column(Text)
    description = Column(Text)

    # Relationships
    video = relationship("Video", back_populates="frames")


# Session management

_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{SQLITE_PATH}",
            echo=False,
            future=True,
        )
        Base.metadata.create_all(_engine)
    return _engine


def get_session() -> "Session":
    """Get a new database session."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal()


def init_db() -> None:
    """Initialize the database (create tables)."""
    engine = get_engine()
    Base.metadata.create_all(engine)


@contextmanager
def managed_session() -> Generator["Session", None, None]:
    """Context manager for database sessions with automatic commit/rollback.

    Usage:
        with managed_session() as session:
            session.add(photo)
            # Auto-commits on success, rolls back on exception

    Yields:
        SQLAlchemy session that auto-commits on context exit.

    Raises:
        Any exception from the session operations (after rollback).
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error("Session rollback due to error: %s", e)
        session.rollback()
        raise
    finally:
        session.close()

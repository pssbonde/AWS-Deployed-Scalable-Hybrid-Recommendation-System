import os
import datetime
from typing import List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from scipy.sparse import load_npz
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sklearn.metrics.pairwise import cosine_similarity

# --- Schemas ---

class SongBase(BaseModel):
    name: str
    artist: str
    spotify_preview_url: str | None = None
    genre: str | None = None
    year: int | None = None

class Song(SongBase):
    id: str

    class Config:
        orm_mode = True

class RecommendationRequest(BaseModel):
    song_name: str
    artist_name: str
    recommender_type: str
    k: int = 10

class UserInteractionBase(BaseModel):
    user_id: str
    song_id: str
    interaction_type: str

class UserInteractionCreate(UserInteractionBase):
    pass

class UserInteraction(UserInteractionBase):
    id: int
    timestamp: datetime.datetime

    class Config:
        orm_mode = True

# --- Database ---

DATABASE_URL = "sqlite:///./music.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBSong(Base):
    __tablename__ = "songs"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    artist = Column(String, index=True)
    spotify_preview_url = Column(String)
    spotify_id = Column(String, index=True)
    tags = Column(String)
    genre = Column(String, index=True)
    year = Column(Integer)
    duration_ms = Column(Integer)
    danceability = Column(Float)
    energy = Column(Float)
    key = Column(Integer)
    loudness = Column(Float)
    mode = Column(Integer)
    speechiness = Column(Float)
    acousticness = Column(Float)
    instrumentalness = Column(Float)
    liveness = Column(Float)
    valence = Column(Float)
    tempo = Column(Float)
    time_signature = Column(Integer)

class DBUserInteraction(Base):
    __tablename__ = "user_interactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    song_id = Column(String, index=True)
    interaction_type = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CRUD ---

def get_songs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBSong).offset(skip).limit(limit).all()

def get_user_interactions(db: Session, user_id: str):
    return db.query(DBUserInteraction).filter(DBUserInteraction.user_id == user_id).all()

def create_user_interaction(db: Session, interaction: UserInteractionCreate):
    db_interaction = DBUserInteraction(**interaction.dict())
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    return db_interaction

# --- Recommender ---

def load_recommendation_data():
    try:
        os.system("ls -l data/")
        return {
            'track_ids': np.load("data/track_ids.npy", allow_pickle=True),
            'collab_filtered_data': pd.read_csv("data/collab_filtered_data.csv"),
            'interaction_matrix': load_npz("data/interaction_matrix.npz"),
        }
    except Exception as e:
        print(f"Error loading recommendation data: {e}")
        return None

def collaborative_recommendation(song_name, artist_name, track_ids, songs_data, interaction_matrix, k=5):
    song_name = song_name.lower()
    artist_name = artist_name.lower()
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    if song_row.empty:
        return pd.DataFrame()
    input_track_id = song_row['track_id'].values.item()
    ind = np.where(track_ids == input_track_id)[0].item()
    input_array = interaction_matrix[ind]
    similarity_scores = cosine_similarity(input_array, interaction_matrix)
    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    recommendation_track_ids = track_ids[recommendation_indices]
    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]
    scores_df = pd.DataFrame({"track_id": recommendation_track_ids.tolist(), "score": top_scores})
    top_k_songs = (
        songs_data
        .loc[songs_data["track_id"].isin(recommendation_track_ids)]
        .merge(scores_df, on="track_id")
        .sort_values(by="score", ascending=False)
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
    )
    return top_k_songs

def get_recommendations(song_name: str, artist_name: str, recommender_type: str, k: int = 10):
    # Mocked recommendations
    return [
        {"id": "mock1", "name": "Dummy Song 1", "artist": "The Mocks", "spotify_preview_url": "https://p.scdn.co/mp3-preview/1", "genre": "mock-rock", "year": 2023},
        {"id": "mock2", "name": "Dummy Song 2", "artist": "The Mocks", "spotify_preview_url": "https://p.scdn.co/mp3-preview/2", "genre": "mock-pop", "year": 2023},
        {"id": "mock3", "name": "Dummy Song 3", "artist": "The Mocks", "spotify_preview_url": "https://p.scdn.co/mp3-preview/3", "genre": "mock-jazz", "year": 2023},
    ]

# --- Database Population ---

def populate_songs():
    create_db_and_tables()
    db = SessionLocal()
    try:
        if db.query(DBSong).count() == 0:
            print("Populating songs table...")
            df = pd.read_csv('data/Music Info.csv')
            df = df.rename(columns={'track_id': 'id'})
            for _, row in df.iterrows():
                song = DBSong(**row.to_dict())
                db.add(song)
            db.commit()
            print("Successfully populated the songs table.")
        else:
            print("Songs table is already populated.")
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

# --- FastAPI App ---

app = FastAPI()

@app.on_event("startup")
def on_startup():
    populate_songs()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/songs/", response_model=List[Song])
def read_songs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    songs = get_songs(db, skip=skip, limit=limit)
    return songs

@app.post("/interactions/", response_model=UserInteraction)
def create_interaction_endpoint(interaction: UserInteractionCreate, db: Session = Depends(get_db)):
    return create_user_interaction(db=db, interaction=interaction)

@app.get("/users/{user_id}/interactions/", response_model=List[UserInteraction])
def read_interactions(user_id: str, db: Session = Depends(get_db)):
    return get_user_interactions(db=db, user_id=user_id)

@app.post("/recommendations/")
def get_recommendations_endpoint(request: RecommendationRequest):
    return get_recommendations(
        song_name=request.song_name,
        artist_name=request.artist_name,
        recommender_type=request.recommender_type,
        k=request.k,
    )

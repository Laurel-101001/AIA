"""
Backend API for AI Contextual Music Recommendation System (OpenAI + Spotify)
===========================================================================
Handles DB initialization, OpenAI API calls, Spotify API calls, and history saving.
"""

import os
import json
import random
import sqlite3
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ─────────────────────────────────────────
# 0. 環境變數載入
# ─────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")

app = FastAPI(title="AI Mood Music API", version="1.0.0")

# ─────────────────────────────────────────
# 1. 資料庫模組
# ─────────────────────────────────────────
DB_NAME = "mood_radio.db"

def init_db() -> None:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            user_mood           TEXT    NOT NULL,
            recommended_tracks  TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    """啟動時確保資料庫已經建立"""
    init_db()

def save_history(user_mood: str, tracks_json: str) -> None:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (timestamp, user_mood, recommended_tracks) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), user_mood, tracks_json),
    )
    conn.commit()
    conn.close()

# ─────────────────────────────────────────
# 2. OpenAI / Spotify 模組
# ─────────────────────────────────────────
SYSTEM_PROMPT = """你是一位專業的音樂情緒分析師與 DJ。
使用者會用自然語言描述他此刻的心情，請你將這段心情轉換為三個 Spotify 音訊特徵數值（範圍 0.0 ~ 1.0）。

你必須嚴格回傳以下 JSON 格式，不可包含任何其他文字：
{"target_valence": 0.XX, "target_energy": 0.XX, "target_acousticness": 0.XX}

各欄位說明：
- target_valence: 正面程度。0 = 極度悲傷，1 = 極度愉悅
- target_energy: 能量高低。0 = 極度平靜，1 = 極度激昂
- target_acousticness: 純音樂程度。0 = 電子/合成，1 = 純原聲"""

def parse_mood_with_openai(user_mood: str) -> dict:
    if not OPENAI_API_KEY:
        raise ValueError("尚未設定 OPENAI_API_KEY")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_mood},
        ],
        temperature=0.7,
        max_tokens=150,
    )
    raw = response.choices[0].message.content
    features = json.loads(raw)

    for key in ("target_valence", "target_energy", "target_acousticness"):
        if key not in features:
            raise ValueError(f"OpenAI 回傳缺少欄位：{key}")
        features[key] = max(0.0, min(1.0, float(features[key])))

    return features

SEED_GENRE_POOL = ["pop", "acoustic", "indie", "r-n-b", "chill", "sad", "happy", "rock"]

def get_spotify_client() -> spotipy.Spotify:
    if not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET:
        raise ValueError("尚未設定 SPOTIPY_CLIENT_ID 與 SPOTIPY_CLIENT_SECRET")
        
    auth_manager = SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
    )
    return spotipy.Spotify(auth_manager=auth_manager)

def recommend_tracks(features: dict, limit: int = 5) -> list:
    sp = get_spotify_client()
    seed = random.choice(SEED_GENRE_POOL)

    results = sp.recommendations(
        seed_genres=[seed],
        target_valence=features["target_valence"],
        target_energy=features["target_energy"],
        target_acousticness=features["target_acousticness"],
        limit=limit,
    )

    tracks = []
    for item in results.get("tracks", []):
        track_id = item["id"]
        try:
            af = sp.audio_features(track_id)
            af_data = af[0] if af else {}
        except Exception:
            af_data = {}

        album_images = item.get("album", {}).get("images", [])
        cover_url = album_images[0]["url"] if album_images else None
        artists = ", ".join(a["name"] for a in item.get("artists", []))

        tracks.append({
            "name": item["name"],
            "artists": artists,
            "cover_url": cover_url,
            "preview_url": item.get("preview_url"),
            "spotify_url": item["external_urls"].get("spotify", ""),
            "valence": af_data.get("valence", features["target_valence"]),
            "energy": af_data.get("energy", features["target_energy"]),
            "acousticness": af_data.get("acousticness", features["target_acousticness"]),
        })

    return tracks


# ─────────────────────────────────────────
# 3. API 路由設定
# ─────────────────────────────────────────
class MoodRequest(BaseModel):
    user_mood: str
    limit: int = 5

@app.post("/api/recommend")
def api_recommend(request: MoodRequest):
    """
    接收心情文字，回傳解析後的特徵數值與 Spotify 推薦歌曲，並存入歷史紀錄。
    """
    mood = request.user_mood.strip()
    if not mood:
        raise HTTPException(status_code=400, detail="請提供有效的心情描述")

    try:
        # 1. 呼叫 OpenAI 取得特徵
        features = parse_mood_with_openai(mood)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI 呼叫失敗: {str(e)}")

    try:
        # 2. 呼叫 Spotify 取得歌曲
        tracks = recommend_tracks(features, limit=request.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spotify 呼叫失敗: {str(e)}")

    # 3. 儲存至資料庫
    if tracks:
        tracks_summary = json.dumps(
            [{"name": t["name"], "artists": t["artists"]} for t in tracks],
            ensure_ascii=False,
        )
        try:
            save_history(mood, tracks_summary)
        except Exception as e:
            # 存檔失敗不阻擋使用者看到推薦結果，僅在 server log 顯示
            print(f"寫入資料庫失敗: {str(e)}")

    return {
        "features": features,
        "tracks": tracks
    }

@app.get("/api/history")
def api_history(limit: int = 10):
    """
    取得推薦歷史紀錄。
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        rows = conn.execute(
            "SELECT timestamp, user_mood, recommended_tracks FROM history ORDER BY id DESC LIMIT ?", 
            (limit,)
        ).fetchall()
        conn.close()
        
        result = []
        for ts, mood, trks in rows:
            try:
                tracks_list = json.loads(trks)
            except:
                tracks_list = []
            
            result.append({
                "timestamp": ts,
                "user_mood": mood,
                "recommended_tracks": tracks_list
            })
        return {"history": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"讀取資料庫失敗: {str(e)}")

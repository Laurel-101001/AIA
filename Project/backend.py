"""
Backend API for AI Contextual Music Recommendation System (Gemini + Pandas + Kaggle Dataset)
===========================================================================
Handles DB initialization, Gemini API calls, local Dataset searching, YTMusic queries, and history saving.
"""

import os
import json
import sqlite3
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# 0. 環境變數載入
# ─────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="AI Mood Music API", version="1.0.0")

# ─────────────────────────────────────────
# 1. 資料庫模組
# ─────────────────────────────────────────
DB_NAME = "mood_radio.db"
DF_SONGS = None

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
    """啟動時確保資料庫已經建立，並載入龐大的 CSV 音樂資料庫"""
    global DF_SONGS
    init_db()
    
    # 載入資料集
    csv_path = "dataset.csv"
    if os.path.exists(csv_path):
        print("🚀 正在載入龐大的 Spotify 音樂資料庫...")
        DF_SONGS = pd.read_csv(csv_path)
        # 清除缺失值並重置 index
        DF_SONGS = DF_SONGS.dropna(subset=['track_name', 'artists', 'valence', 'energy', 'acousticness', 'track_genre'])
        print(f"✅ 成功載入 {len(DF_SONGS)} 首歌曲！")
    else:
        print("⚠️ 警告：找不到 dataset.csv！推薦系統將無法正常運作。")

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
# 2. Gemini / Pandas / YTMusic 模組
# ─────────────────────────────────────────
SYSTEM_PROMPT = """你是一位專業的音樂情緒分析師與 DJ。
使用者會用自然語言描述他此刻的心情，請你將這段心情轉換為三個 Spotify 音訊特徵數值（範圍 0.0 ~ 1.0），以及一個目標曲風分類（target_genre）。

請從以下曲風中挑選出一個最適合的 target_genre：
acoustic, afrobeat, alt-rock, alternative, ambient, anime, black-metal, bluegrass, blues, bossanova, brazil, breakbeat, british, cantopop, chicago-house, children, chill, classical, club, comedy, country, dance, dancehall, death-metal, deep-house, detroit-techno, disco, disney, drum-and-bass, dub, dubstep, edm, electro, electronic, emo, folk, forro, french, funk, garage, german, gospel, goth, grindcore, groove, grunge, guitar, happy, hard-rock, hardcore, hardstyle, heavy-metal, hip-hop, holidays, honky-tonk, house, idm, indian, indie, indie-pop, industrial, iranian, j-dance, j-idol, j-pop, j-rock, jazz, k-pop, kids, latin, latino, malay, mandopop, metal, metal-misc, metalcore, minimal-techno, movies, mpb, new-age, new-release, opera, pagode, party, philippines, piano, pop, pop-film, post-dubstep, power-pop, progressive-house, psych-rock, punk, punk-rock, r-n-b, rainy-day, reggae, reggaeton, road-trip, rock, rock-n-roll, rockabilly, romance, sad, salsa, samba, sertanejo, show-tunes, singer-songwriter, ska, sleep, songwriter, soul, soundtracks, spanish, study, summer, swedish, synth-pop, tango, techno, trance, trip-hop, turkish, work-out, world-music

你必須嚴格回傳以下 JSON 格式，不可包含任何其他文字：
{
  "target_valence": 0.XX,
  "target_energy": 0.XX,
  "target_acousticness": 0.XX,
  "target_genre": "字詞"
}

各欄位說明：
- target_valence: 正面程度。0 = 極度悲傷，1 = 極度愉悅
- target_energy: 能量高低。0 = 極度平靜，1 = 極度激昂
- target_acousticness: 純音樂程度。0 = 電子/合成，1 = 純原聲"""

def parse_mood_with_gemini(user_mood: str) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("尚未設定 GEMINI_API_KEY")

    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_PROMPT)
    
    response = model.generate_content(
        user_mood,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.7
        )
    )

    raw = response.text
    features = json.loads(raw)

    for key in ("target_valence", "target_energy", "target_acousticness"):
        if key not in features:
            raise ValueError(f"Gemini 回傳缺少欄位：{key}")
        features[key] = max(0.0, min(1.0, float(features[key])))
        
    if "target_genre" not in features:
        features["target_genre"] = "pop" # fallback

    return features

def recommend_tracks(features: dict, limit: int = 5) -> list:
    global DF_SONGS
    if DF_SONGS is None or DF_SONGS.empty:
        raise ValueError("尚未載入 dataset.csv，無法進行推薦！請確定已重啟後端。")

    # 1. 根據曲風篩選資料 (如果找不到該曲風，就用全部資料)
    genre = features["target_genre"]
    df_filtered = DF_SONGS[DF_SONGS["track_genre"] == genre]
    if df_filtered.empty:
        df_filtered = DF_SONGS

    # 2. 計算歐幾里得距離 (Euclidean Distance)
    # distance = sqrt((valence-tv)^2 + (energy-te)^2 + (acousticness-ta)^2)
    # 我們可以省略 sqrt 開根號，直接比較平方和即可，效率更高
    df_filtered = df_filtered.copy()
    
    v_diff = df_filtered["valence"] - features["target_valence"]
    e_diff = df_filtered["energy"] - features["target_energy"]
    a_diff = df_filtered["acousticness"] - features["target_acousticness"]
    
    df_filtered["distance"] = v_diff**2 + e_diff**2 + a_diff**2
    
    # 3. 排序並取前 N 首
    # 去除重複歌名的歌曲
    df_filtered = df_filtered.drop_duplicates(subset=['track_name', 'artists'])
    
    top_songs = df_filtered.nsmallest(limit, "distance")
    
    # 4. 準備回傳的資料
    tracks = []
    
    for _, row in top_songs.iterrows():
        track_name = row["track_name"]
        artists = row["artists"]
        track_id = row["track_id"]
        
        tracks.append({
            "name": track_name,
            "artists": artists,
            "cover_url": None,
            "preview_url": None,
            "youtube_url": None,
            "spotify_url": f"https://open.spotify.com/track/{track_id}",
            "spotify_embed": f"https://open.spotify.com/embed/track/{track_id}",
            "valence": float(row["valence"]),
            "energy": float(row["energy"]),
            "acousticness": float(row["acousticness"]),
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
    接收心情文字，回傳解析後的特徵數值與推薦歌曲，並存入歷史紀錄。
    """
    mood = request.user_mood.strip()
    if not mood:
        raise HTTPException(status_code=400, detail="請提供有效的心情描述")

    try:
        features = parse_mood_with_gemini(mood)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini 呼叫失敗: {str(e)}")

    try:
        tracks = recommend_tracks(features, limit=request.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset / YTMusic 推薦失敗: {str(e)}")

    if tracks:
        tracks_summary = json.dumps(
            [{"name": t["name"], "artists": t["artists"]} for t in tracks],
            ensure_ascii=False,
        )
        try:
            save_history(mood, tracks_summary)
        except Exception as e:
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

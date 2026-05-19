"""
🎵 AI 情境音樂推薦系統 (純前端 UI)
=========================================
透過 HTTP requests 向 FastAPI 後端取得音樂推薦資料。
"""

import json
import streamlit as st
import plotly.graph_objects as go
import requests

API_BASE_URL = "http://127.0.0.1:8000"

def make_radar_chart(track: dict) -> go.Figure:
    """為單首歌曲繪製 valence / energy / acousticness 雷達圖。"""
    categories = ["Valence（愉悅）", "Energy（能量）", "Acousticness（原聲）"]
    values = [track["valence"], track["energy"], track["acousticness"]]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                name=track["name"],
                line_color="#1DB954",       # Spotify 綠
                fillcolor="rgba(29,185,84,0.25)",
            )
        ]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=280,
    )
    return fig


def main():
    st.set_page_config(page_title="AI 情境音樂推薦", page_icon="🎵", layout="wide")

    st.title("🎵 AI 情境音樂推薦系統：你說心情，我懂你")
    st.caption("這是一個純前端介面，透過 API 呼叫後端引擎為您挑歌。")

    user_mood = st.text_input(
        "描述你的心情 ✍️",
        placeholder="例如：期中考超爛我超想大哭＋尖叫",
    )

    submitted = st.button("🎶 幫我推薦音樂！", type="primary", use_container_width=True)

    if submitted and user_mood.strip():
        with st.spinner("🚀 正在請後端大腦分析與尋找歌曲..."):
            try:
                # 呼叫後端 API
                response = requests.post(
                    f"{API_BASE_URL}/api/recommend",
                    json={"user_mood": user_mood, "limit": 5},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    features = data.get("features", {})
                    tracks = data.get("tracks", [])
                    
                    st.success(
                        f"✅ 情緒解析完成！ Valence={features.get('target_valence', 0):.2f}　"
                        f"Energy={features.get('target_energy', 0):.2f}　"
                        f"Acousticness={features.get('target_acousticness', 0):.2f}"
                    )
                    
                    if not tracks:
                        st.warning("⚠️ 找不到符合的歌曲，請換個心情描述試試！")
                    else:
                        st.markdown("---")
                        st.subheader("🎧 為你推薦的歌曲")

                        for idx, track in enumerate(tracks):
                            with st.container():
                                col_img, col_info, col_chart = st.columns([1, 2, 2])

                                with col_img:
                                    if track.get("cover_url"):
                                        st.image(track["cover_url"], width=160)

                                with col_info:
                                    st.markdown(f"### {idx + 1}. {track['name']}")
                                    st.markdown(f"**🎤 {track['artists']}**")
                                    if track.get("spotify_url"):
                                        st.markdown(f"[🔗 在 Spotify 收聽]({track['spotify_url']})")
                                    if track.get("preview_url"):
                                        st.audio(track["preview_url"], format="audio/mp3")
                                    else:
                                        st.info("此歌曲暫無 30 秒試聽片段")

                                with col_chart:
                                    fig = make_radar_chart(track)
                                    st.plotly_chart(fig, use_container_width=True)

                            st.markdown("---")
                            
                else:
                    error_msg = response.json().get('detail', '未知錯誤')
                    st.error(f"❌ 後端 API 錯誤：{error_msg}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ 無法連線至後端。請確定 FastAPI 已經在 {API_BASE_URL} 上啟動！")
            except Exception as e:
                st.error(f"❌ 發生未預期的錯誤：{e}")

    elif submitted:
        st.warning("⚠️ 請先輸入你的心情再按推薦！")

    # ---- 側邊欄：歷史紀錄 ----
    with st.sidebar:
        st.header("📜 推薦歷史紀錄")
        try:
            res = requests.get(f"{API_BASE_URL}/api/history", params={"limit": 10}, timeout=5)
            if res.status_code == 200:
                history = res.json().get("history", [])
                if history:
                    for item in history:
                        ts = item.get("timestamp", "")
                        mood = item.get("user_mood", "")
                        trks = item.get("recommended_tracks", [])
                        
                        with st.expander(f"🕐 {ts[:16]}"):
                            st.markdown(f"**心情：** {mood}")
                            for t in trks:
                                st.markdown(f"- 🎵 {t.get('name')}（{t.get('artists')}）")
                else:
                    st.info("還沒有任何紀錄，快來試試看！")
            else:
                st.error("⚠️ 無法取得歷史紀錄")
        except requests.exceptions.ConnectionError:
            st.info("尚未連線到後端，歷史紀錄暫不提供。")


if __name__ == "__main__":
    main()

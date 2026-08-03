from __future__ import annotations

import json

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Smart Retail AI", page_icon="🛍️", layout="wide")
st.title("🛍️ Smart Retail & Customer Intelligence Platform")
st.caption("Frontend for the FastAPI services defined in the major-project document.")

api_base = st.sidebar.text_input("API base URL", "http://localhost:8000").rstrip("/")
api_key = st.sidebar.text_input("API key", "dev-secret-key", type="password")
headers = {"X-API-Key": api_key}


def show_response(response: requests.Response):
    try:
        payload = response.json()
    except ValueError:
        st.error(response.text)
        return
    if response.ok:
        st.json(payload)
    else:
        st.error(payload)


tabs = st.tabs(["Analytics", "Sentiment", "Chatbot", "Product", "Face Recognition"])

with tabs[0]:
    if st.button("Refresh analytics"):
        try:
            response = requests.get(f"{api_base}/dashboard/stats", headers=headers, timeout=20)
            if response.ok:
                data = response.json()
                cols = st.columns(4)
                cols[0].metric("Face visits", data["total_face_visits"])
                cols[1].metric("Recognized", data["recognized_visits"])
                cols[2].metric("Unknown", data["unknown_visits"])
                cols[3].metric("Unique customers", data["unique_returning_customers"])
                for title, key in [
                    ("Sentiment", "sentiment_distribution"),
                    ("Products", "product_distribution"),
                    ("Chatbot intents", "top_chatbot_intents"),
                ]:
                    st.subheader(title)
                    series = pd.Series(data[key], dtype="int64")
                    if series.empty:
                        st.info("No predictions logged yet.")
                    else:
                        st.bar_chart(series)
                st.subheader("Latest activity")
                st.dataframe(data["latest_activity"], use_container_width=True)
            else:
                show_response(response)
        except requests.RequestException as exc:
            st.error(str(exc))

with tabs[1]:
    text = st.text_area("Customer review", "The product quality is excellent and delivery was fast.")
    if st.button("Analyze sentiment"):
        try:
            show_response(requests.post(f"{api_base}/analyze-sentiment", headers=headers, json={"text": text}, timeout=20))
        except requests.RequestException as exc:
            st.error(str(exc))

with tabs[2]:
    message = st.text_input("Message", "What is your return policy?")
    if st.button("Ask chatbot"):
        try:
            show_response(requests.post(f"{api_base}/chatbot", headers=headers, json={"message": message}, timeout=20))
        except requests.RequestException as exc:
            st.error(str(exc))

with tabs[3]:
    product_image = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg"], key="product")
    if product_image and st.button("Classify product"):
        try:
            files = {"image": (product_image.name, product_image.getvalue(), product_image.type)}
            show_response(requests.post(f"{api_base}/classify-product", headers=headers, files=files, timeout=30))
        except requests.RequestException as exc:
            st.error(str(exc))

with tabs[4]:
    st.warning("Use only images from people who explicitly consented to this classroom demonstration.")
    face_image = st.file_uploader("Upload a frontal face image", type=["png", "jpg", "jpeg"], key="face")
    if face_image and st.button("Recognize face"):
        try:
            files = {"image": (face_image.name, face_image.getvalue(), face_image.type)}
            show_response(requests.post(f"{api_base}/recognize-face", headers=headers, files=files, timeout=30))
        except requests.RequestException as exc:
            st.error(str(exc))

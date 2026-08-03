# System architecture

```text
Client / Streamlit / Postman / webcam image
                     |
                     | REST + X-API-Key
                     v
                 FastAPI
       /recognize-face  /classify-product
       /analyze-sentiment  /chatbot
              /dashboard/stats
          |          |           |
          v          v           v
    CV Service   NLP Service   Chatbot Service
    OpenCV       TF-IDF        Rules + ML intent
    face DB      Logistic Reg. FAQ retrieval
          \          |          /
           \         |         /
            Unified pipeline
                   |
                   v
        SQLite privacy-minimized logs
```

`app/pipeline.py` constructs every service once. FastAPI stores the pipeline in application state during startup, avoiding model reloads on each request.

## Storage

The database contains aggregate-friendly event records:

- `customer_visits`
- `sentiment_logs`
- `chat_logs`
- `product_predictions`

Raw review/chat text and uploaded images are not stored. Text is represented by a SHA-256 hash in logs, and face registration stores a numerical embedding in `face_db.pkl`.

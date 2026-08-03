# API examples

Start the server first:

```bash
uvicorn app.main:app --reload
```

The default development API key is `dev-secret-key`. Change it with `SMART_RETAIL_API_KEY`.

## Sentiment

```bash
curl -X POST http://localhost:8000/analyze-sentiment \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d '{"text":"The product is excellent and delivery was fast."}'
```

## Chatbot

```bash
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d '{"message":"What is your return policy?"}'
```

## Product classification

```bash
curl -X POST http://localhost:8000/classify-product \
  -H "X-API-Key: dev-secret-key" \
  -F "image=@product.jpg"
```

## Face recognition

```bash
curl -X POST http://localhost:8000/recognize-face \
  -H "X-API-Key: dev-secret-key" \
  -F "image=@consenting_customer.jpg"
```

## Dashboard statistics

```bash
curl http://localhost:8000/dashboard/stats \
  -H "X-API-Key: dev-secret-key"
```

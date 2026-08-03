# Consenting face samples only

Do not place scraped or non-consenting face images here. For a classroom demo, use your own images or images from volunteers who understand the purpose and can request deletion.

Recommended registration route:

```bash
curl -X POST http://localhost:8000/register-face \
  -H "X-API-Key: dev-secret-key" \
  -F "customer_id=CUST001" \
  -F "customer_name=Demo User" \
  -F "consent=true" \
  -F "image=@consenting_photo.jpg"
```

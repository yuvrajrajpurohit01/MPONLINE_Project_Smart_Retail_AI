# Recommended dataset links

The repository includes small synthetic/sample data only so it remains lightweight and starts without a large download. Use one of these sources for evaluated training:

## Product classification

- TensorFlow Fashion-MNIST: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist
- RPC (Retail Product Checkout) official site: https://rpc-dataset.github.io/
- RPC Kaggle mirror: https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset

## Face-recognition research practice

- Scikit-learn LFW loader: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html

LFW is appropriate for research practice, not proof of consent for identifying shoppers. For a retail classroom demo, prefer self-collected images from informed volunteers.

## Review sentiment

- Women's E-Commerce Clothing Reviews: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

Map ratings 1–2 to Negative, 3 to Neutral, and 4–5 to Positive only if that labeling rule is documented in your report.

## Chatbot

- `data/intents.json` is original project data with 25 FAQ intents and can be expanded for the selected retailer.

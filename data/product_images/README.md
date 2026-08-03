# Product image data

The included `demo/` images are synthetic colour-and-shape examples used only to make the API runnable immediately. They are not a real retail dataset and should not be used to report accuracy.

For a real model, organize images as:

```text
data/product_images/train/
├── bags/
├── clothing/
├── electronics/
├── groceries/
└── shoes/
```

Then run:

```bash
pip install -r requirements-ml.txt
python scripts/train_product_classifier.py --data-dir data/product_images/train --epochs 8
```

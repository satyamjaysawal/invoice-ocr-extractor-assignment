import os
import shutil
from dotenv import load_dotenv

load_dotenv()

kaggle_token = os.getenv("KAGGLE_TOKEN")
if not kaggle_token:
    raise ValueError("KAGGLE_TOKEN not found in .env file. See .env.example for format.")

os.environ["KAGGLE_TOKEN"] = kaggle_token

import kagglehub

print("Downloading dataset from Kaggle (this may take a few minutes)...")
path = kagglehub.dataset_download("osamahosamabdellatif/high-quality-invoice-images-for-ocr")
print(f"Dataset downloaded to: {path}")

source_dir  = os.path.join(path, "batch_1", "batch_1", "batch1_1")
script_dir  = os.path.dirname(os.path.abspath(__file__))
invoices_dir = os.path.join(script_dir, "invoices")
os.makedirs(invoices_dir, exist_ok=True)

copied = 0
for i in range(331, 382):
    fname = f"batch1-{i:04d}.jpg"
    src   = os.path.join(source_dir, fname)
    dst   = os.path.join(invoices_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        print(f"  WARNING: {fname} not found in dataset")

print(f"\nDone! Copied {copied} invoice images to: {invoices_dir}")
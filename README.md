# Invoice OCR Extractor

Extracts structured data from invoice images using Pytesseract (Tesseract v5) and OpenCV. Processes 51 invoice images and saves results to CSV and Excel.

---

## Project Structure

```
28_May_Invoice_Assessment/
├── extract_invoices.py      # Main OCR extraction script
├── download_dataset.py      # Kaggle dataset downloader
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── invoices/                # Input invoice images (not tracked in git)
├── output.csv               # Extracted data (CSV)
└── output.xlsx              # Extracted data (Excel)
```

---

## Extracted Fields

| Field | Description |
|---|---|
| Filename | Source image name |
| Seller Name | Name of the seller |
| Seller Tax ID | Seller's tax ID (XXX-XX-XXXX) |
| Client Name | Name of the client |
| Client Tax ID | Client's tax ID (XXX-XX-XXXX) |
| Invoice Number | Unique invoice number |
| Invoice Date | Date of issue |
| Net Worth | Pre-tax total |
| VAT | Tax amount (10%) |
| Gross Worth | Final total (Net + VAT) |

---

## Setup

### 1. Install Tesseract OCR

Download and install from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) or use winget:

```bash
winget install UB-Mannheim.TesseractOCR
```

Default install path: `C:\Program Files\Tesseract-OCR\`

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle API token

1. Go to [kaggle.com](https://www.kaggle.com) → Account → **Create New Token**
2. A `kaggle.json` file will download containing your credentials:
   ```json
   {"username": "your_username", "key": "your_api_key"}
   ```
3. Copy your username and key into a `.env` file in the project root:
   ```
   KAGGLE_TOKEN={"username":"your_username","key":"your_api_key"}
   ```

### 4. Download invoice images

```bash
python download_dataset.py
```

This downloads the dataset from Kaggle and copies images `batch1-0331.jpg` to `batch1-0381.jpg` into the `invoices/` folder automatically.

> Alternatively, place invoice images manually inside the `invoices/` folder and skip this step.

---

## Usage

```bash
python extract_invoices.py
```

Output files are saved in the same directory:
- `output.csv`
- `output.xlsx`

---

## How It Works

```
Invoice Image
     │
     ▼
Preprocessing        →   Grayscale, upscale, adaptive threshold
     │
     ▼
Tesseract OCR        →   Full text + per-word bounding boxes
     │
     ▼
Field Extraction     →   Regex patterns + spatial layout analysis
     │
     ▼
Validation           →   Checks Net + VAT ≈ Gross (±2%), auto-corrects order
     │
     ▼
output.csv + output.xlsx
```

---

## Results

| Metric | Value |
|---|---|
| Total invoices | 51 |
| Successfully processed | 51 |
| Errors | 0 |
| Average time per invoice | ~1.8 seconds |
| Total runtime | ~1 min 31 sec |

---

## Dependencies

| Package | Purpose |
|---|---|
| pytesseract | Python wrapper for Tesseract OCR |
| Pillow | Image loading |
| opencv-python | Image preprocessing |
| pandas | DataFrame and Excel export |
| openpyxl | Excel file writer |
| kagglehub | Kaggle dataset downloader |

---

## Dataset

[High Quality Invoice Images for OCR](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr) — Kaggle
Images used: `batch1-0331.jpg` to `batch1-0381.jpg` (51 images)

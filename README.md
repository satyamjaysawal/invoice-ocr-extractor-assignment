# Invoice OCR Extractor

**Extract structured fields from invoice images using Tesseract OCR, OpenCV preprocessing, regex, and spatial layout heuristics.**

This project processes a batch of 51 high-quality invoice images and reliably extracts seller/client details, invoice metadata, and financial totals into clean CSV and Excel outputs.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Extracted Data Schema](#extracted-data-schema)
- [How It Works (Pipeline)](#how-it-works-pipeline)
- [Key Algorithms & Techniques](#key-algorithms--techniques)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results & Output](#results--output)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Limitations & Notes](#limitations--notes)

---

## Overview

This is a practical computer vision + NLP engineering project that demonstrates a complete end-to-end OCR extraction system for semi-structured documents (invoices).

The core challenge is that invoices have **variable layouts**. Pure regex on raw OCR text often fails because:
- Field labels and values can be positioned differently
- Numbers can be misread or misordered
- "Total" lines contain three related values that must be correctly assigned

The solution combines:
1. Careful image preprocessing for better OCR accuracy
2. Both full-text OCR **and** word-level bounding box data (`image_to_data`)
3. Spatial reasoning (positions of "Seller", "Client", "Tax Id" labels)
4. Multiple regex fallbacks
5. A clever validation step that tries permutations of monetary values to satisfy the invariant `Net + VAT ≈ Gross`

---

## Features

- Downloads a specific slice of the Kaggle invoice dataset automatically
- Robust preprocessing (grayscale + conditional upscaling + adaptive thresholding)
- Dual OCR path: full text string + spatially-aware word boxes
- Intelligent seller/client name detection using label position + image midpoint
- Proximity-based Tax ID extraction
- Multi-strategy total amount parsing (regex on "Total ..." line + spatial fallback)
- **Smart monetary validation**: uses `itertools.permutations` to find the ordering where `Net + VAT ≈ Gross` (within 2% relative error)
- Resume-safe processing (skips already-processed files in `output.csv`)
- Dual output: `output.csv` + `output.xlsx`
- Zero hard dependencies on ML models (pure classical CV + rules)

---

## Project Structure

```
invoice-ocr-extractor-assignment/
├── README.md                 # This file - complete project documentation
├── extract_invoices.py       # Main extraction engine (OCR + parsing + validation)
├── download_dataset.py       # One-time Kaggle dataset downloader (uses .env)
├── requirements.txt          # All Python dependencies
├── .env.example              # Template for Kaggle credentials
├── .gitignore
├── invoices/                 # 51 source invoice images (batch1-0331.jpg ... batch1-0381.jpg)
│   └── *.jpg
├── output.csv                # Structured extraction results
└── output.xlsx               # Same data in Excel format
```

> Note: `invoices/`, `output.*`, and `.env` are git-ignored (except the example).

---

## Extracted Data Schema

| Column           | Type   | Description                              | Example             |
|------------------|--------|------------------------------------------|---------------------|
| Filename         | string | Original image filename                  | batch1-0331.jpg     |
| Seller Name      | string | Company/person issuing the invoice       | Ochoa-Scott         |
| Seller Tax ID    | string | Seller tax identifier                    | 921-82-1053         |
| Client Name      | string | Customer / billed party                  | Green LLC           |
| Client Tax ID    | string | Client tax identifier (may be empty)     | 965-99-1263         |
| Invoice Number   | string | Unique invoice identifier                | 94138597            |
| Invoice Date     | string | Date of issue (as printed)               | 02/05/2018          |
| Net Worth        | string | Pre-tax subtotal (preserves original formatting) | 1612,50        |
| VAT              | string | Tax amount (typically 10%)               | 161,25              |
| Gross Worth      | string | Final total = Net + VAT                  | 1773,75             |

All monetary values keep the original European comma-decimal format present in the source images.

---

## How It Works (Pipeline)

```
Invoice Image (JPG)
        │
        ▼
┌───────────────────────┐
│   Preprocess Image    │  grayscale, optional 2x upscale, Gaussian blur,
└───────────┬───────────┘  adaptive Gaussian threshold
            │
            ▼
┌─────────────────────────────────────────────┐
│  Tesseract OCR (two calls)                  │
│  • image_to_string  → full_text             │
│  • image_to_data    → list of (x,y,w,h,text,conf) │
└───────────┬─────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  extract_fields(full_text, spatial_items)              │
│  ├─ Invoice Number & Date via regex on full text       │
│  ├─ Seller/Client labels → find vertical candidates    │
│  │   below label, left/right of image midpoint         │
│  ├─ Tax IDs via spatial proximity + regex fallback     │
│  └─ Net/VAT/Gross via "Total ..." regex or spatial     │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  validate_and_fix_summary()                            │
│  Try all 6 permutations of the 3 numbers.              │
│  Accept first where (Net + VAT) ≈ Gross within 2%.     │
│  Fallback: sort by magnitude (VAT smallest).           │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
   Append row → output.csv  (+ pandas → output.xlsx)
```

---

## Key Algorithms & Techniques

### 1. Image Preprocessing (`preprocess_image`)
- Convert to grayscale
- If both dimensions < 2000 px → 2× cubic upscaling (helps Tesseract on smaller scans)
- Light Gaussian blur
- `cv2.adaptiveThreshold` with `ADAPTIVE_THRESH_GAUSSIAN_C` — excellent for uneven lighting common in photos of paper invoices

### 2. Spatial Item Representation
Word bounding boxes are turned into simple `(x, y, text, conf)` tuples. This allows:
- "Find text to the right of label X within Y tolerance"
- "Find text below label X"
- Midpoint-based left/right column assignment for Seller vs Client

### 3. Seller / Client Name Detection
- Locate the exact tokens `"Seller"` and `"Client"` (case-insensitive, exact match after stripping colon)
- Use their x-positions to determine which is left column (Seller) vs right column (Client)
- Collect non-numeric, non-label text in the 80-pixel vertical band immediately below each label
- Join words that share approximately the same y-coordinate

### 4. Monetary Value Disambiguation (`validate_and_fix_summary`)
This is the most interesting part of the project.

Many invoices print a "Total" row containing three numbers in varying visual order:
```
Net     VAT     Gross
```

Sometimes OCR reads them left-to-right, sometimes the layout makes the order ambiguous.

The function:
- Parses the three candidate strings into floats
- Tries **all 6 permutations**
- Scores each by `|Net + VAT - Gross| / Gross`
- Picks the permutation with relative error < 2% **and** satisfying `VAT < Net <= Gross`
- If none qualify, falls back to sorting by size (VAT smallest, Gross largest)

This single function dramatically improves extraction quality without needing training data.

### 5. Resume / Idempotency
`main()` reads any existing `output.csv`, builds a set of already-processed filenames, and only runs OCR on the remaining images. This is very useful during development and when re-running the script.

---

## Setup & Installation

### Prerequisites

**Tesseract OCR v5** (required)

Windows (recommended):
```powershell
winget install UB-Mannheim.TesseractOCR
```

The script hardcodes the path:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Other OS: install via package manager and update the path in `extract_invoices.py:10`.

### Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Kaggle Credentials (only needed for first download)

1. Create a Kaggle account and go to **Account** → "Create New Token"
2. This downloads `kaggle.json`
3. Create a `.env` file in the project root:

```env
KAGGLE_TOKEN="your_kaggle_api_key_here"
```

> The token value is just the `key` field from kaggle.json (not the whole JSON).

### Download the Images

```powershell
python download_dataset.py
```

This will:
- Download the full Kaggle dataset via `kagglehub`
- Copy exactly the 51 images `batch1-0331.jpg` through `batch1-0381.jpg` into `./invoices/`

You can also manually place any `.jpg` files named in that range into `invoices/`.

---

## Usage

```powershell
python extract_invoices.py
```

The script will:
1. Verify Tesseract is available
2. Locate the 51 images (either in `./invoices/` or in the kagglehub cache)
3. Process any not-yet-seen images
4. Append results to `output.csv`
5. Regenerate `output.xlsx` from the full CSV

Typical runtime: ~1.5–2 minutes for all 51 images on a modern laptop.

---

## Results & Output

**Final result on the provided batch:**

| Metric                    | Value          |
|---------------------------|----------------|
| Total invoices            | 51             |
| Successfully processed    | 51             |
| Images with errors        | 0              |
| Missing Client Tax ID     | 1 (batch1-0378) | 
| Rows in output.csv        | 51             |

Sample rows (first and last):

```
Filename         Seller Name              Invoice #   Net Worth   Gross Worth
batch1-0331.jpg  Ochoa-Scott              94138597    1612,50     1773,75
...
batch1-0381.jpg  Williams PLC             76272465     604,99      665,49
```

Both `output.csv` and `output.xlsx` are committed in the repo for convenience (they contain no secrets).

---

## Dependencies

| Package           | Purpose                              |
|-------------------|--------------------------------------|
| pytesseract       | Python interface to Tesseract        |
| Pillow            | Image handling for Tesseract         |
| opencv-python     | Preprocessing (resize, threshold)    |
| pandas            | DataFrame + Excel export             |
| openpyxl          | Excel (.xlsx) writer engine          |
| python-dotenv     | Load `KAGGLE_TOKEN` from .env        |
| kagglehub         | Download dataset without kaggle CLI  |

---

## Dataset

**Source:** [High Quality Invoice Images for OCR](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr) on Kaggle (by osamahosamabdellatif)

**Subset used:** Images `batch1-0331.jpg` to `batch1-0381.jpg` (51 images from `batch_1/batch1_1`).

These are clean, high-resolution synthetic or scanned invoices with consistent but varied layouts — ideal for testing OCR extraction logic.

---

## Limitations & Notes

- **Windows-centric**: Tesseract path is hardcoded for Windows.
- **English-only** invoices assumed (the dataset is English).
- No deep learning / layout models (e.g. no LayoutLM, Donut, etc.). Pure classical approach.
- Name extraction can occasionally pick up stray words if the layout is unusual (see row 3 in sample data).
- Dates and invoice numbers are taken as strings exactly as OCR'd.
- The monetary validation assumes VAT rate ≈ 10% and that `Net <= Gross`. It will still work for other small tax rates as long as the three numbers are present.
- The project deliberately favors **deterministic, explainable rules** over black-box models.

### Potential Improvements

- Make Tesseract path configurable via environment variable
- Add confidence thresholding / per-field quality scores
- Support more international date and number formats
- Add a small Streamlit or Gradio demo UI
- Export to JSON Lines or a database

---

## License & Credits

This project was created for a DSA / practical assignment focused on OCR engineering and text/spatial data processing.

Special thanks to the Kaggle dataset author for providing clean invoice images suitable for benchmarking extraction pipelines.

---

**To explore the implementation, start with [extract_invoices.py](/extract_invoices.py) — particularly:**
- `preprocess_image`
- `extract_fields`
- `validate_and_fix_summary`
- `process_invoice`

These four functions contain the majority of the interesting logic.

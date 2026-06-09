import os
import re
import csv
import cv2
import numpy as np
import pytesseract
from PIL import Image
from itertools import permutations

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TESSERACT_CONFIG = "--oem 3 --psm 6"


def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 2000:
        scale = 2
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    return binary


def ocr_image_full_text(image_path: str) -> str:
    processed = preprocess_image(image_path)
    pil_img = Image.fromarray(processed)
    return pytesseract.image_to_string(pil_img, config=TESSERACT_CONFIG)


def ocr_image_data(image_path: str) -> list:
    processed = preprocess_image(image_path)
    pil_img = Image.fromarray(processed)
    data = pytesseract.image_to_data(
        pil_img,
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT
    )
    items = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if text and conf > 10:
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            items.append((x, y, w, h, text, conf))
    return items


def build_spatial_items(ocr_data: list) -> list:
    return [(x, y, text, conf) for (x, y, w, h, text, conf) in ocr_data]


def find_label_value(spatial_items, label_pattern, direction="right", y_tol=20, x_tol=500):
    for x, y, text, conf in spatial_items:
        if re.search(label_pattern, text, re.IGNORECASE):
            if direction == "right":
                candidates = [
                    (sx, sy, st)
                    for sx, sy, st, sc in spatial_items
                    if abs(sy - y) < y_tol and sx > x and sx - x < x_tol
                ]
                candidates.sort(key=lambda c: c[0])
                if candidates:
                    return " ".join(c[2] for c in candidates)
            elif direction == "below":
                candidates = [
                    (sx, sy, st)
                    for sx, sy, st, sc in spatial_items
                    if sy > y and sy - y < 80 and abs(sx - x) < 300
                ]
                if candidates:
                    candidates.sort(key=lambda c: c[1])
                    return candidates[0][2]
    return ""


def clean_number(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"[^\d,\.]", "", s)
    s = s.strip(".,")
    return s


def parse_number(s: str) -> float:
    s = clean_number(s)
    if not s:
        return 0.0
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) == 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def extract_fields(full_text: str, spatial_items: list) -> dict:
    fields = {
        "Seller Name": "",
        "Seller Tax ID": "",
        "Client Name": "",
        "Client Tax ID": "",
        "Invoice Number": "",
        "Invoice Date": "",
        "Net Worth": "",
        "VAT": "",
        "Gross Worth": ""
    }

    for pat in [
        r"Invoice\s*[Nn]o[\.:\s=]+\s*(\d+)",
        r"Invoice\s*#\s*(\d+)",
        r"Inv\s*[Nn]o[\.:\s=]+\s*(\d+)",
    ]:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            fields["Invoice Number"] = m.group(1).strip()
            break

    for pat in [
        r"Date\s*of\s*issue[:\s='\"]+\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
        r"Issue\s*[Dd]ate[:\s]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
        r"Date[:\s]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
        r"(\d{2}/\d{2}/\d{4})",
        r"(\d{2}\.\d{2}\.\d{4})",
    ]:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            fields["Invoice Date"] = m.group(1).strip()
            break

    seller_label = None
    client_label = None
    img_midpoint_x = 0

    for x, y, text, conf in spatial_items:
        text_clean = text.strip().rstrip(":").strip()
        if re.match(r"^Seller$", text_clean, re.IGNORECASE):
            seller_label = (x, y)
        if re.match(r"^Client$", text_clean, re.IGNORECASE):
            client_label = (x, y)
        img_midpoint_x = max(img_midpoint_x, x)

    img_midpoint_x = img_midpoint_x / 2

    if seller_label and client_label:
        seller_pos = seller_label if seller_label[0] < client_label[0] else client_label
        client_pos = client_label if seller_label[0] < client_label[0] else seller_label
    else:
        seller_pos = seller_label
        client_pos = client_label

    if seller_pos:
        sx, sy = seller_pos
        candidates = []
        for ix, iy, it, ic in spatial_items:
            if (iy > sy and iy - sy < 80 and iy - sy > 5
                    and ix < img_midpoint_x + 50
                    and not re.search(r"(Tax Id|IBM|Client|Seller|Invoice|Date|No\.|Qty|UM|Description)", it, re.IGNORECASE)
                    and not re.match(r"^\d", it)):
                candidates.append((ix, iy, it))
        candidates.sort(key=lambda c: (c[1], c[0]))
        if candidates:
            first_y = candidates[0][1]
            name_parts = [c[2] for c in candidates if abs(c[1] - first_y) < 20]
            fields["Seller Name"] = " ".join(name_parts).strip().rstrip(",;:")

    if client_pos:
        cx, cy = client_pos
        candidates = []
        for ix, iy, it, ic in spatial_items:
            if (iy > cy and iy - cy < 80 and iy - cy > 5
                    and ix > img_midpoint_x - 180
                    and not re.search(r"(Tax Id|IBM|Client|Seller|Invoice|Date|No\.|Qty|UM|Description)", it, re.IGNORECASE)
                    and not re.match(r"^\d", it)):
                candidates.append((ix, iy, it))
        candidates.sort(key=lambda c: (c[1], c[0]))
        if candidates:
            first_y = candidates[0][1]
            name_parts = [c[2] for c in candidates if abs(c[1] - first_y) < 20]
            fields["Client Name"] = " ".join(name_parts).strip().rstrip(",;:")

    tax_entries = []
    for x, y, text, conf in spatial_items:
        if re.search(r"Tax\s*Id", text, re.IGNORECASE):
            for ix, iy, it, ic in spatial_items:
                if abs(iy - y) < 25 and ix - x > 0 and ix - x < 200 and re.match(r"[\d][\d\-]+[\d]", it):
                    tax_entries.append((x, y, it.strip().rstrip(",")))
                    break

    tax_ids_regex = re.findall(r"Tax\s*Id[:\s]*([\d\-]+[\d])", full_text, re.IGNORECASE)
    tax_ids = [t[2] for t in sorted(tax_entries, key=lambda t: t[0])]
    if len(tax_ids) < 2 and tax_ids_regex:
        tax_ids = tax_ids_regex if len(tax_ids_regex) >= len(tax_ids) else tax_ids

    if len(tax_ids) >= 2:
        fields["Seller Tax ID"] = tax_ids[0]
        fields["Client Tax ID"] = tax_ids[1]
    elif len(tax_ids) == 1:
        fields["Seller Tax ID"] = tax_ids[0]

    dollar_pattern = r"\$?\s*([\d\s,\.]*)"
    m = re.search(
        r"Total\s+" + dollar_pattern + r"\s+\$?\s*" + dollar_pattern + r"\s+\$?\s*" + dollar_pattern,
        full_text, re.IGNORECASE
    )
    if m:
        fields["Net Worth"]   = clean_number(m.group(1))
        fields["VAT"]         = clean_number(m.group(2))
        fields["Gross Worth"] = clean_number(m.group(3))
    else:
        lines = full_text.split("\n")
        for line in lines:
            if re.search(r"Total", line, re.IGNORECASE) and "$" in line:
                nums = re.findall(r"\$?\s*([\d\s,\.]+)", line)
                if len(nums) >= 3:
                    fields["Net Worth"]   = clean_number(nums[0])
                    fields["VAT"]         = clean_number(nums[1])
                    fields["Gross Worth"] = clean_number(nums[2])
                    break

    if not fields["Net Worth"]:
        for x, y, text, conf in spatial_items:
            if re.match(r"^Total", text, re.IGNORECASE):
                totals = []
                for ix, iy, it, ic in spatial_items:
                    if abs(iy - y) < 25 and ix > x and re.search(r"[\d]", it):
                        totals.append((ix, it.strip("$").strip()))
                totals.sort(key=lambda t: t[0])
                nums = [clean_number(t[1]) for t in totals if clean_number(t[1])]
                if len(nums) >= 3:
                    fields["Net Worth"]   = nums[0]
                    fields["VAT"]         = nums[1]
                    fields["Gross Worth"] = nums[2]
                    break

    return fields


def validate_and_fix_summary(fields: dict) -> dict:
    net_s   = fields.get("Net Worth", "")
    vat_s   = fields.get("VAT", "")
    gross_s = fields.get("Gross Worth", "")
    if not (net_s and vat_s and gross_s):
        return fields

    net   = parse_number(net_s)
    vat   = parse_number(vat_s)
    gross = parse_number(gross_s)
    values    = [net, vat, gross]
    originals = [net_s, vat_s, gross_s]

    best_perm = None
    best_diff = float("inf")
    for perm in permutations(range(3)):
        n, v, g = values[perm[0]], values[perm[1]], values[perm[2]]
        if n <= 0:
            continue
        diff     = abs((n + v) - g)
        rel_diff = diff / g if g != 0 else float("inf")
        if rel_diff < 0.02 and v < n <= g:
            if rel_diff < best_diff:
                best_diff = rel_diff
                best_perm = perm

    if best_perm is not None:
        fields["Net Worth"]   = originals[best_perm[0]]
        fields["VAT"]         = originals[best_perm[1]]
        fields["Gross Worth"] = originals[best_perm[2]]
    else:
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        fields["VAT"]         = originals[indexed[0][0]]
        fields["Net Worth"]   = originals[indexed[1][0]]
        fields["Gross Worth"] = originals[indexed[2][0]]

    return fields


def process_invoice(image_path: str) -> dict:
    filename = os.path.basename(image_path)
    try:
        full_text     = ocr_image_full_text(image_path)
        ocr_data      = ocr_image_data(image_path)
        spatial_items = build_spatial_items(ocr_data)
        fields        = extract_fields(full_text, spatial_items)
        fields        = validate_and_fix_summary(fields)
        fields["Filename"] = filename
        return fields, None
    except Exception as e:
        fields = {
            "Filename":       filename,
            "Seller Name":    "",
            "Seller Tax ID":  "",
            "Client Name":    "",
            "Client Tax ID":  "",
            "Invoice Number": "",
            "Invoice Date":   "",
            "Net Worth":      "",
            "VAT":            "",
            "Gross Worth":    "",
        }
        return fields, str(e)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dir  = os.path.join(script_dir, "invoices")

    if os.path.isdir(local_dir) and any(f.endswith(".jpg") for f in os.listdir(local_dir)):
        dataset_base = local_dir
    else:
        dataset_base = os.path.join(
            os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
            "osamahosamabdellatif", "high-quality-invoice-images-for-ocr",
            "versions", "3", "batch_1", "batch_1", "batch1_1"
        )

    image_files = []
    for i in range(331, 382):
        fname = f"batch1-{i:04d}.jpg"
        fpath = os.path.join(dataset_base, fname)
        if os.path.exists(fpath):
            image_files.append(fpath)
    image_files.sort()

    if not image_files:
        print("ERROR: No images found. Check the dataset path.")
        return

    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        print(f"ERROR: Tesseract not found: {e}")
        return

    output_path = os.path.join(script_dir, "output.csv")
    fieldnames  = [
        "Filename", "Seller Name", "Seller Tax ID", "Client Name", "Client Tax ID",
        "Invoice Number", "Invoice Date", "Net Worth", "VAT", "Gross Worth"
    ]

    processed_files = set()
    all_results     = []

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                reader_csv = csv.DictReader(f)
                for row in reader_csv:
                    if row.get("Filename"):
                        processed_files.add(row["Filename"])
                        all_results.append(row)
        except Exception:
            processed_files, all_results = set(), []

    remaining = [f for f in image_files if os.path.basename(f) not in processed_files]
    file_mode = "a" if (os.path.exists(output_path) and processed_files) else "w"

    with open(output_path, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()
            f.flush()

        for img_path in remaining:
            result, _ = process_invoice(img_path)
            writer.writerow(result)
            f.flush()
            all_results.append(result)

    try:
        import pandas as pd
        excel_path = os.path.join(script_dir, "output.xlsx")
        df = pd.DataFrame(all_results, columns=fieldnames)
        df.to_excel(excel_path, index=False, sheet_name="Invoices")
    except ImportError:
        pass

    print(f"Done. {len(all_results)} invoices saved to {output_path}")


if __name__ == "__main__":
    main()
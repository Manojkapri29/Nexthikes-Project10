import streamlit as st
import cv2
import pytesseract
import re
import os
import io
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image

#  Configure Tesseract path (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Custom OCR Lab Report Extractor", layout="wide")
st.title(" Custom OCR Lab Report Extractor (YOLO + Tesseract)")
st.write("Upload images and/or YOLO annotation .txt files. If annotation exists, crops will be OCR'd per-field.")

# 📤 File uploader
uploaded_files = st.file_uploader(
    "Upload images and/or YOLO .txt files (or mixed)",
    type=["jpg", "jpeg", "png", "txt"],
    accept_multiple_files=True
)

# Helper: save uploads temporarily
def save_temp_file(uploaded_file):
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

#  Text Extraction Function
def extract_fields_from_text(full_text: str) -> dict:
    text = re.sub(r'\s+', ' ', full_text).strip()
    res = {
        "Patient Name": "Not found",
        "Age/Sex": "Not found",
        "Sample ID": "Not found",
        "Test Asked": "Not found",
        "T3": "Not found",
        "T4": "Not found",
        "TSH": "Not found"
    }

    # Patient Name
    m = re.search(r'NAME\s*[:+\-]?\s*([A-Z][A-Z\s]{2,60}?)\s*(?:\(|SAMPLE|TEST)', text)
    if m:
        candidate = m.group(1).strip()
        bad_tokens = ['TECHNOLOGY', 'VALUE', 'UNITS', 'REFERENCE', 'RANGE', 'TOTAL']
        if not any(bt.lower() in candidate.lower() for bt in bad_tokens):
            res["Patient Name"] = candidate.title()

    # Age / Sex
    m = re.search(r'\((\d{1,3})\s*[Yy]?\s*[\/]?\s*([MmFf])?\)', text)
    if m:
        age = m.group(1)
        sex = m.group(2).upper() if m.group(2) else ""
        res["Age/Sex"] = f"{age}/{sex}" if sex else f"{age}"

    # Sample ID
    m = re.search(r'[xX]\s*\(?\s*(\d{6,12})\s*\)?', text)
    if m:
        res["Sample ID"] = m.group(1)
    else:
        m = re.search(r'(?:sample|lab\s*no|ref)[^\d]{0,10}([0-9]{4,12})', text, re.I)
        if m:
            res["Sample ID"] = m.group(1)

    # Test Name
    m = re.search(r'TEST\s*(?:ASKED|NAME)?\s*[:\-]?\s*([A-Z][A-Z0-9\s\-]{2,40}?)(?=\s+(?:TECHNOLOGY|VALUE|UNITS|REFERENCE|RANGE|TOTAL|$))', text, re.I)
    if m:
        tn = m.group(1).strip(" :.-")
    res["Test Asked"] = tn.title()


    # Test Values (T3, T4, TSH)
    t3 = re.search(r'TRIIODOTHYRONINE[^\d]*([0-9]{1,3}(?:\.[0-9]+)?)', text, re.I)
    if t3: res["T3"] = t3.group(1)
    t4 = re.search(r'THYROXINE[^\d]*([0-9]{1,3}(?:\.[0-9]+)?)', text, re.I)
    if t4: res["T4"] = t4.group(1)
    tsh = re.search(r'THYROID\s*STIMULATING\s*HORMONE[^\d]*([0-9]{1,3}(?:\.[0-9]+)?)', text, re.I)
    if tsh: res["TSH"] = tsh.group(1)

    # Cleanup
    if "TECHNOLOGY" in res["Patient Name"]:
        res["Patient Name"] = "Not found"
    return res

#  OCR Function
def extract_text_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# PDF Generator
def generate_pdf(data_dict, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFont("Helvetica", 14)
    c.drawString(100, 800, "OCR Report Summary")
    c.setFont("Helvetica", 12)
    y = 770
    for key, val in data_dict.items():
        c.drawString(100, y, f"{key}: {val}")
        y -= 25
    c.save()

#  Main processing
if uploaded_files:
    results = []
    for file in uploaded_files:
        st.markdown(f"##  Processing: {file.name}")
        file_path = save_temp_file(file)

        # Detect image or annotation
        if file.name.lower().endswith((".jpg", ".jpeg", ".png")):
            text = extract_text_from_image(file_path)
        elif file.name.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        st.subheader("Extracted Raw Text:")
        st.text(text[:2000])  # preview

        fields = extract_fields_from_text(text)
        st.subheader("Extracted Fields")
        st.json(fields)
        results.append(fields)

        # Save PDF for each file
        os.makedirs("outputs", exist_ok=True)
        pdf_path = f"outputs/ocr_report_{file.name}.pdf"
        generate_pdf(fields, pdf_path)

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name=f"ocr_report_{file.name}.pdf",
                key=f"download_{file.name}"
            )

    # Optional: combined table
    if len(results) > 1:
        df = pd.DataFrame(results)
        st.subheader("Combined Results Table")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download All Results as CSV",
            data=csv,
            file_name="ocr_results.csv",
            key="csv_all"
        )

else:
    st.info("Please upload at least one image or YOLO .txt annotation file.")

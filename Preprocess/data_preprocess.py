import os
import json

import pandas as pd
import pytesseract
import pdfplumber
from tqdm import tqdm
from pdf2image import convert_from_path
from multiprocessing import Pool


# This function from original code of bm25_retrieve.py
def read_pdf(pdf_loc, page_infos: list = None):
    """Read PDF file and return text content"""
    pdf_text = ""
    with pdfplumber.open(pdf_loc) as pdf:

        pages = pdf.pages[page_infos[0] : page_infos[1]] if page_infos else pdf.pages

        for page in pages:
            text = page.extract_text()
            if text:
                pdf_text += text

    return pdf_text


def remove_color(image, threshold=130):
    """Convert the color of images to white, except for black"""
    image = image.convert("RGBA")
    data = image.getdata()
    new_data = []
    for item in data:
        if item[0] > threshold:
            new_data.append((255, 255, 255, item[3]))  # 替換為白色
        else:
            new_data.append(item)  # 保留原來的顏色
    image.putdata(new_data)
    # 將圖片轉換為 RGB 以儲存為 JPEG
    image = image.convert("RGB")
    return image


def extract_text(image):
    """Extract text from image using Tesseract OCR"""
    text = pytesseract.image_to_string(image, lang="chi_tra", config="--psm 6")
    text = text.replace("\n", "").replace(" ", "")
    return text


def pdf_image_label_extractor(path, dpi=300):
    """Extract label of PDF file using OCR"""
    # 讀取 PDF 檔案，取得第一頁的圖片
    pages = convert_from_path(path, dpi=dpi)
    image = pages[0]
    # 裁切圖片
    width, height = image.size
    image = image.crop((0, 0, width, height // 4))
    # 去除顏色
    # image.save(f"{path[:-4]}_origin.jpg")
    image = remove_color(image)
    # image.save(f"{path[:-4]}_removed.jpg")
    text = extract_text(image)
    return text[:50]


def pdf_image_whole_file(path, dpi=300):
    """Extract text from whole PDF file using OCR"""
    pages = convert_from_path(path, dpi=dpi)
    text = str()
    for i, image in enumerate(pages):
        image = remove_color(image)
        text += extract_text(image)
    return text


def read_pdf_image(path):
    """Read PDF file and return text content using OCR"""
    pages = convert_from_path(path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="chi_tra")

    return text


def preprocessor(path, reference=["finance", "insurance"]):
    """Text preprocessor
    If the text is empty, use OCR to extract text from the image
    Otherwise, use the text directly
    """
    print("Preprocessor")
    for ref in reference:
        ref_path = os.path.join(path, ref)
        index = []
        labels = []
        texts = []
        i = 0
        pdf_files = [file for file in os.listdir(ref_path) if file.endswith(".pdf")]
        for file_name in tqdm(pdf_files, desc=f"Processing {ref} files"):
            file_path = os.path.join(ref_path, file_name)
            text = read_pdf(file_path)
            text = text.replace("\n", "").replace("\r", "").replace(" ", "")
            if text == "":
                text = read_pdf_image(file_path)
                text = text.replace("\n", "").replace("\r", "").replace(" ", "")
                labels.append(pdf_image_label_extractor(file_path, dpi=400))
            else:
                labels.append(text[:50])

            # Remove characters that might interfere with JSON parsing
            index.append(file_name[:-4])
            texts.append(text)

        data = [
            {"index": idx, "label": lbl, "text": txt}
            for idx, lbl, txt in zip(index, labels, texts)
        ]
        with open(f"./{path}/{ref}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def process_file_ocr(file_path):
    """Subfunction for multiprocessing of preprocessor_ocr"""
    text = pdf_image_whole_file(file_path, dpi=400)
    text = text.replace("\n", "").replace("\r", "").replace("\f", "").replace(" ", "")

    text = text[50:]
    label = text[:50]
    idx = os.path.basename(file_path)[:-4]  # 更改變數名稱
    return idx, label, text


def preprocessor_ocr(file_path, reference=["finance"], workers=1):
    """OCR preprocessor
    Use OCR to extract text from images for all PDF files
    """
    print("Preprocessor_ocr")
    for ref in reference:
        ref_path = os.path.join(file_path, ref)
        pdf_files = [
            os.path.join(ref_path, file)
            for file in os.listdir(ref_path)
            if file.endswith(".pdf")
        ]

        with Pool(workers) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(process_file_ocr, pdf_files),
                total=len(pdf_files),
                desc=f"Processing {ref} files",
            ):
                results.append(result)

        data = [{"index": idx, "label": lbl, "text": txt} for idx, lbl, txt in results]
        with open(f"{file_path}/ocr_{ref}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

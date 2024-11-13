# 2024 玉山人工智慧公開挑戰賽 - 初賽資格審查

## 資料夾結構

```
 1 .
 2 ├ Preprocess
 3 │ ├ corpus_preprocessor.py
 4 │ └ data_preprocess.py
 5 ├ Model
 6 │ └ retrievers.py
 7 ├ dataset
 8 │ └ preliminary
 9 │   └ questions_preliminary.json
10 ├ reference
11 │ ├ faq
12 │ │ └ pid_map_context.json
13 │ ├ finance.json
14 │ └ insurance.json
15 ├ README.md
16 └ requirements.txt
```

## 資料夾說明

### Preprocess

#### corpus_preprocessor.py

#### data_preprocess.py

There two methods to preprocess the raw data.

1. `preprocesor`: Extract the text directly from pdf file, if content is empty, use pytesseract OCR to extract the text.
2. `preprocesor_ocr`: Extract the text by using pytesseract OCR for all raw pdf file.

- The raw data will put in `./reference` direactory to read and save the preprocessed data.

```python
    from Preprocess.data_preprocess import preprocessor, preprocessor_ocr

    file_path = "./reference"
    preprocessor(file_path)
    preprocessor_ocr(file_path)
```

### Model

#### retrievers.py

### dataset

### reference

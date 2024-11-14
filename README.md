# 2024 玉山人工智慧公開挑戰賽 - 初賽資格審查

## Requirements

我們這組的環境設置如下：

- OS: Ubuntu 22.04
- Python: 3.9.20
- Virtual environment: [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- CUDA: 12.4

在初賽中，我們這組的虛擬環境是由 micromamba 所建構。若需要還原我們的環境設定會需要先安裝好 micromamba，並在 CUDA 12.4 之下利用 shell 執行：

```bash
{your_shell} install.sh
```

這段腳本會自動由`enviroment.yml`建立一個新的 micromamba 虛擬環境，環境名稱為 **ai_cup_env**，接著再自動下載`requirements.txt`中的 packages。

## 前處理流程

我們在前處理的過程中，主要分為三個階段：

1. 將官方提供的`finance`, `insurance`中所有的 pdf 轉為 json 儲存。詳見底下`資料夾說明 - Preprocess - data_preprocess.py`的說明。
2. 觀察儲存好的`finance.json`, `insurance.json`，以人工的方式再進行一次前處理。主要包含：
   a. 取出前 50 個字作為該文件的 label
   b. 將年號改為西元年
   c. 刪除所有因轉置而導致無法辨識的文件刪除，僅留存標題與日期
3. 利用人工處理好後的資料集，在程式將其讀入記憶體、準備檢索前，會決定是否要對文本進行 chuncking 以及文本的 rewrite。

## 採用的 Retriever 模型

我們在初賽中主要利用了兩個模型，分別為 **Okapi BM25**, **jina-reranker-v2-base-multilingual** [[Link](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)] 。其中，我們在初賽中提出的最高分的模型（第二次 attempt）設置為：

```python
25 # 1-2. Set Hyperparameters
26 rewrite = {
27      "query": True,
28      "text": True,
29  }
30  # If we don't want to chunk, just set "activate" to False
31  chunking = {
32      "activate": True,
33      "chunk_size": {"insurance": 128, "finance": 256, "jina_default": 1024},
34      "overlap_size": {"insurance": 32, "finance": 32, "jina_default": 80},
35  }
36  # Each value should be either "full" or int
37  top_k = {"BM25": 5, "jina": 1}
```

( **Note**: 該設置可在`main.py`中看到。 )

其中：

- `rewrite["query"]`: 為 bool 值，用來決定 query 是否需要進行 rewrite。
- `rewrite["text"]`: 為 bool 值，用來決定文本是否需要進行 rewrite。
- `chunking["activate"]`: 為 bool 值，用來決定文本是否需要進行 chunking。
- `chunking["chunk_size"]`: 為字典，用來決定 insurnace, finance 每個 chunk 的 size，以及 **jina-reranker** 中`model.rerank`這個方法中所需的參數。
- `chunking["overlap_size"]`: 為字典，用來決定 insurnace, finance 每個 chunk 間的 overlap 大小，以及 **jina-reranker** 中`model.rerank`這個方法中所需的參數。
- `top_k["BM25"]`: 為 int 值或`"full"`。若輸入 int 值 $k$ ，則在利用`BM25-first`此函數時，會檢索與問題最相關的 top-$k$文本。若輸入`"full"`，則會將該問題 source 中所有相關的文本都檢索出來。
- `top_k["jina"]`: 與`top_k["BM25"]`的概念相同。

## 資料夾結構

```
 1 .
 2 ├ Preprocess
 3 │ ├ data_preprocess.py
 4 │ └ corpus_preprocessor.py
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
16 ├ requirements.txt
17 ├ environment.yml
18 └ install.sh
```

## 資料夾說明

### Preprocess

這個資料夾中包含兩個模組，分別為`data_preprocess.py`與`corpus_preprocessor.py`。前者是我們用來將官方提供的資料集進行第一次前處理的模組；後者是我們用來將前處理過後的資料集讀入記憶體中、以及進行第三次前處理的模組。

#### `data_preprocess.py`

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

#### `corpus_preprocessor.py`

這個模組主要會利用到兩個函數：

1. `load_corpus_to_df`:
   主要會使用到的引數有：

   - `category`: 決定我們目前要讀取什麼類別的資料。
   - `source_path`: 儲存該類別所有語料的 json 檔路徑。
   - `chuncking`: 判斷是否需要對語料庫進行 chuncking，以及各個類別的 chunk 大小、chunk 間重疊的大小。
   - `rewrite`: 判斷是否對每個文本進行改寫。若要改寫，會針對季度、日期將中文數字統一改為阿拉伯數字。

   最終的輸出會是一個 pd.DataFrame，包含兩個屬性：

   - `id`: 各個文本的 id
   - `text`: 準備送入 retriever 的乾淨文本

   **Note:** `faq`類別的資料中，一個 id 下若有$m$個問題、每個問題分別有$n_1,\dots,n_m$種回答，最終會在 DataFrame 中為這個 id 建立$n_1+\cdots+n_m$個項目。

2. `load_queries`:
   會使用到的引數有:

   - `source_path`: 儲存所有問題的 json 檔路徑
   - `rewrite`: 判斷是否對每個文本進行改寫。若要改寫，會針對季度、日期將中文數字統一改為阿拉伯數字，同時也會將公司簡稱改寫為全稱。

   最終的輸出會是一個 list，每個元素皆為包含 qid, source, query, category 這些鍵的字典。

### Model

這個資料夾中只包含`retrievers.py`這個模組，其功能是將處理好的資料集通過 retriever 來獲取與各個 query 最相關的幾個文本。

#### `retrievers.py`

這個模組主要會利用到`BM25_first`這個函數，在這個函數中我們會先利用 **BM25** 對各個問題與 source 中的文本進行比對，並檢索出分數前五高的文本。隨後，會再將這五個文本利用 **jina-reranker-v2-base-multilingual** 檢索出與問題最相關的文本。
其主要會用到的引數有：

- `queries`: 接收`load_queries`產生的 queries list。
- `insurance_corpus`: 接收`load_corpus_to_df`對 insurance 類別產生的 DataFrame
- `finance_corpus`: 接收`load_corpus_to_df`對 finance 類別產生的 DataFrame
- `faq_corpus`: 接收`load_corpus_to_df`對 faq 類別產生的 DataFrame
- `chunking`: 依照先前是否有對語料庫進行 chunking，來決定如何使用 **jina-reranker** 。若有 chunking，會使用`model.rerank`的方法來計算文本與問題的相關性分數；若沒有，則會利用`model.compute_score`的方法來計算。
- `top_k`: 決定 **BM25**, **jina-reranker** 分別需要檢索出多少最相關的文本。

### dataset

包含官方所提供的初賽問題集`questions_preliminary.json`。

### reference

包含官方所提供的資料集`faq/pid_map_context.json`、對 insurance 和 finance 的資料進行第一次前處理、人工處理的 json 檔`finance.json`, `insurance.json`。

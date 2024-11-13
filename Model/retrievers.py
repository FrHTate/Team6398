"""
This module contains the functions to retrieve documents from the corpus by BM25 and Jina.\n
There are four functions in this module:\n
1. BM25_retrieve: Retrieve top-k documents by BM25 for each query.\n
2. jina_retrieve: Retrieve top-k documents by Jina for each query.\n
3. BM25_first: Retrieve top-k documents by BM25 first, then retrieve top-1 by Jina.\n
4. jina_first: Retrieve top-k documents by Jina first, then retrieve top-1 by BM25.\n
"""

# General
import pandas as pd
from tqdm import tqdm

# For BM25
# from ckiptagger import WS
import jieba
from rank_bm25 import BM25Okapi

# For Jina
import torch
from transformers import AutoModelForSequenceClassification

# Define the function that could be accessed by other files
__all__ = [
    "BM25_retrieve",
    "jina_retrieve",
    "BM25_first",
    "jina_first",
]

# Load jina reranker
model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype="auto",
    trust_remote_code=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)


def BM25_retrieve(
    queries, insurance_corpus, finance_corpus, faq_corpus, tokenizer="jieba", top_k=5
):
    """
    Retrieve top-k documents by BM25 for each query.

    Args:
        queries (list): List of queries. Each query is a dictionary with keys: "qid", "source", "query", "category".
        insurance_corpus (pd.DataFrame): Dataframe of insurance corpus. Each row is a document with attributes: "id", "text".
        finance_corpus (pd.DataFrame): Dataframe of finance corpus. Each row is a document with attributes: "id", "text".
        faq_corpus (pd.DataFrame): Dataframe of faq corpus. Each row is a document with attributes: "id", "text".
        tokenizer (str): Tokenizer to use. Default is "jieba".
        top_k (int): Number of documents to retrieve for each query. Default is 5.

    Returns:
        pd.DataFrame: Dataframe of retrieved documents. Each row is the top-k documents for each query with attributes: "id", "text", "score".
    """
    id_list = []
    text_list = []
    score_list = []
    for query in tqdm(queries, desc="Retrieving by BM25"):
        if query["category"] == "insurance":
            source_corpus = insurance_corpus[
                insurance_corpus["id"].isin(query["source"])
            ]
        elif query["category"] == "finance":
            source_corpus = finance_corpus[finance_corpus["id"].isin(query["source"])]
        elif query["category"] == "faq":
            source_corpus = faq_corpus[faq_corpus["id"].isin(query["source"])]
        else:
            raise ValueError("Missing category")

        if tokenizer == "jieba":
            tokenized_query = list(jieba.cut_for_search(query["query"]))
            tokenized_corpus = [
                list(jieba.cut_for_search(doc)) for doc in source_corpus["text"]
            ]
        elif tokenizer == "ckiptagger":
            raise NotImplementedError("CKIP is not implemented yet")
        else:
            raise ValueError("Invalid tokenizer")

        bm25 = BM25Okapi(tokenized_corpus)

        scores = bm25.get_scores(tokenized_query)
        sorted_indice = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        if top_k != "full":
            sorted_indice = sorted_indice[:top_k]

        id_list.append([source_corpus.iloc[i]["id"] for i in sorted_indice])
        text_list.append([source_corpus.iloc[i]["text"] for i in sorted_indice])
        score_list.append([scores[i] for i in sorted_indice])

    return pd.DataFrame({"id": id_list, "text": text_list, "score": score_list})


def jina_retrieve(
    queries,
    insurance_corpus,
    finance_corpus,
    faq_corpus,
    chunking={"activate": False, "chunk_size": None, "overlap_size": None},
    top_k=5,
):
    """
    Retrieve top-k documents by Jina for each query.

    Args:
        queries (list): List of queries. Each query is a dictionary with keys: "qid", "source", "query", "category".
        insurance_corpus (pd.DataFrame): Dataframe of insurance corpus. Each row is a document with attributes: "id", "text".
        finance_corpus (pd.DataFrame): Dataframe of finance corpus. Each row is a document with attributes: "id", "text".
        faq_corpus (pd.DataFrame): Dataframe of faq corpus. Each row is a document with attributes: "id", "text".
        chunking (dict): a dictionary containing the chunking configuration, should have the following keys\n
            "activate": bool, whether to chunk the text\n
            "chunk_size": dict, containing the chunk size for each category\n
            "overlap_size": dict, containing the overlap size for each category
        top_k (int): Number of documents to retrieve for each query. Default is 5.

    Returns:
        pd.DataFrame: Dataframe of retrieved documents. Each row is the top-k documents for each query with attributes: "id", "text", "score".
    """
    id_list = []
    text_list = []
    score_list = []

    for query in tqdm(queries, desc="Retrieving by Jina"):
        if query["category"] == "insurance":
            source_corpus = insurance_corpus[
                insurance_corpus["id"].isin(query["source"])
            ]
        elif query["category"] == "finance":
            source_corpus = finance_corpus[finance_corpus["id"].isin(query["source"])]
        elif query["category"] == "faq":
            source_corpus = faq_corpus[faq_corpus["id"].isin(query["source"])]
        else:
            raise ValueError("Missing category")

        if not chunking["activate"]:  # Use rerank function in Jina for each document
            if query["category"] == "insurance":
                chunk_size = chunking["chunk_size"]["insurance"]
                overlap_size = chunking["overlap_size"]["insurance"]

            elif query["category"] == "finance":
                chunk_size = chunking["chunk_size"]["finance"]
                overlap_size = chunking["overlap_size"]["finance"]

            else:  # Default, for faq
                chunk_size = chunking["chunk_size"]["jina_default"]
                overlap_size = chunking["overlap_size"]["jina_default"]

            if top_k == "full":
                top_k = len(source_corpus)
            # result = [{'document': str, 'relevance_score': float, 'index': int}, ...] with reverse order (high -> low)
            result = model.rerank(
                query["query"],
                source_corpus["text"].tolist(),
                max_query_length=chunk_size // 2,  # at most half of max_query_length
                max_length=chunk_size,
                overlap=overlap_size,
                top_n=top_k,
            )

            id_list.append([source_corpus.iloc[doc["index"]]["id"] for doc in result])
            text_list.append([doc["document"] for doc in result])
            score_list.append([doc["relevance_score"] for doc in result])

            # top5_id = [int(doc["id"]) for  in result]

        else:  # Use function compute_score in Jina for each chunk
            sentence_pairs = [
                [query["query"], chunk] for chunk in source_corpus["text"]
            ]

            scores = model.compute_score(sentence_pairs, max_length=1024)

            sorted_indice = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            if top_k != "full":
                sorted_indice = sorted_indice[:top_k]

            id_list.append([source_corpus.iloc[i]["id"] for i in sorted_indice])
            text_list.append([source_corpus.iloc[i]["text"] for i in sorted_indice])
            score_list.append([scores[i] for i in sorted_indice])

    return pd.DataFrame({"id": id_list, "text": text_list, "score": score_list})


def _BM25_retrieve_series(queries, topk_df, tokenizer="jieba", top_k=5):
    """
    If the first retrieval is done by Jina, then retrieve the top-k documents by BM25 for each query.

    Args:
        queries (list): List of queries. Each query is a dictionary with keys: "qid", "source", "query", "category".
        topk_df (pd.DataFrame): Dataframe of top-k documents retrieved by Jina. Each row is the top-k documents for each query with attributes: "id", "text", "score".
        tokenizer (str): Tokenizer to use. Default is "jieba".
        top_k (int): Number of documents to retrieve for each query. Default is 5.

    Returns:
        pd.DataFrame: Dataframe of retrieved documents. Each row is the top-k documents for each query with attributes: "id", "text", "score".
    """
    id_list = []
    text_list = []
    score_list = []
    for i, query in enumerate(tqdm(queries, desc="Retrieving by BM25")):
        source_corpus = topk_df.iloc[i]["text"]

        if tokenizer == "jieba":
            tokenized_query = list(jieba.cut_for_search(query["query"]))
            tokenized_corpus = [
                list(jieba.cut_for_search(doc)) for doc in source_corpus
            ]
        elif tokenizer == "ckiptagger":
            raise NotImplementedError("CKIP is not implemented yet")

        bm25 = BM25Okapi(tokenized_corpus)

        scores = bm25.get_scores(tokenized_query)
        sorted_indice = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        if top_k != "full":
            sorted_indice = sorted_indice[:top_k]

        id_list.append([topk_df.iloc[i]["id"][j] for j in sorted_indice])
        text_list.append([topk_df.iloc[i]["text"][j] for j in sorted_indice])
        score_list.append([scores[j] for j in sorted_indice])

    return pd.DataFrame({"id": id_list, "text": text_list, "score": score_list})


def _jina_retrieve_series(queries, topk_df, chunking={"activate": False}, top_k=5):
    """
    If the first retrieval is done by BM25, then retrieve the top-k documents by Jina for each query.

    Args:
        queries (list): List of queries. Each query is a dictionary with keys: "qid", "source", "query", "category".
        topk_df (pd.DataFrame): Dataframe of top-k documents retrieved by BM25. Each row is the top-k documents for each query with attributes: "id", "text", "score".
        chunking (dict): a dictionary containing the chunking configuration, should have the following keys\n
            "activate": bool, whether to chunk the text\n
            "chunk_size": dict, containing the chunk size for each category\n
            "overlap_size": dict, containing the overlap size for each category
        top_k (int): Number of documents to retrieve for each query. Default is 5.

    Returns:
        pd.DataFrame: Dataframe of retrieved documents. Each row is the top-k documents for each query with attributes: "id", "text", "score".
    """
    id_list = []
    text_list = []
    score_list = []
    for i, query in enumerate(tqdm(queries, desc="Retrieving by Jina")):
        source_corpus = topk_df.iloc[i]["text"]

        if not chunking["activate"]:
            if query["category"] == "insurance":
                chunk_size = chunking["chunk_size"]["insurance"]
                overlap_size = chunking["overlap_size"]["insurance"]

            elif query["category"] == "finance":
                chunk_size = chunking["chunk_size"]["finance"]
                overlap_size = chunking["overlap_size"]["finance"]

            else:  # Default, for faq
                chunk_size = chunking["chunk_size"]["jina_default"]
                overlap_size = chunking["overlap_size"]["jina_default"]

            result = model.rerank(
                query["query"],
                source_corpus,
                max_query_length=chunk_size // 2,  # at most half of max_query_length
                max_length=chunk_size,
                overlap=overlap_size,
                top_n=top_k,
            )

            id_list.append([topk_df.iloc[i]["id"][doc["index"]] for doc in result])
            text_list.append([doc["document"] for doc in result])
            score_list.append([doc["relevance_score"] for doc in result])

        else:
            sentence_pairs = [[query["query"], chunk] for chunk in source_corpus]

            scores = model.compute_score(sentence_pairs, max_length=1024)

            sorted_indice = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            if top_k != "full":
                sorted_indice = sorted_indice[:top_k]

            id_list.append([topk_df.iloc[i]["id"][j] for j in sorted_indice])
            text_list.append([topk_df.iloc[i]["text"][j] for j in sorted_indice])
            score_list.append([scores[j] for j in sorted_indice])

    return pd.DataFrame({"id": id_list, "text": text_list, "score": score_list})


def BM25_first(
    queries,
    insurance_corpus,
    finance_corpus,
    faq_corpus,
    tokenizer="jieba",
    chunking={"activate": False, "chunk_size": None, "overlap_size": None},
    top_k={"BM25": 5, "jina": 1},
):
    """
    Retrieve top-k documents by BM25 first, then retrieve top-1 by Jina.

    Args:
        queries (list): List of queries. Each query is a dictionary with keys: "qid", "source", "query", "category".
        insurance_corpus (pd.DataFrame): Dataframe of insurance corpus. Each row is a document with attributes: "id", "text".
        finance_corpus (pd.DataFrame): Dataframe of finance corpus. Each row is a document with attributes: "id", "text".
        faq_corpus (pd.DataFrame): Dataframe of faq corpus. Each row is a document with attributes: "id", "text".
        tokenizer (str): Tokenizer for BM25 to use. Default is "jieba".
        chunking (dict): a dictionary containing the chunking configuration, should have the following keys\n
            "activate": bool, whether to chunk the text\n
            "chunk_size": dict, containing the chunk size for each category\n
            "overlap_size": dict, containing the overlap size for each category
        top_k (dict): Number of documents to retrieve for each query. Default is 5.

    Returns:
        tuple (pd.DataFrame, pd.DataFrame):
            1. Dataframe of retrieved documents. Each row is the top-1 document for each query with attributes: "id", "text", "score".\n
            2. Dataframe of retrieved documents by BM25 only. Each row is the top-k documents for each query with attributes: "id", "text", "score".
    """
    df_BM25_result = BM25_retrieve(
        queries,
        insurance_corpus,
        finance_corpus,
        faq_corpus,
        tokenizer=tokenizer,
        top_k=top_k["BM25"],
    )

    df_jina_result = _jina_retrieve_series(
        queries, df_BM25_result, chunking=chunking, top_k=top_k["jina"]
    )

    return df_jina_result, df_BM25_result


def jina_first(
    queries,
    insurance_corpus,
    finance_corpus,
    faq_corpus,
    tokenizer="jieba",
    chunking={"activate": False, "chunk_size": None, "overlap_size": None},
    top_k={"BM25": 1, "jina": 5},
):
    """
    Retrieve top-k documents by Jina first, then retrieve top-1 by BM25.

    Args:
        queries (list): List of queries. Each query is a dictionary with keys: "qid", "source", "query", "category".
        insurance_corpus (pd.DataFrame): Dataframe of insurance corpus. Each row is a document with attributes: "id", "text".
        finance_corpus (pd.DataFrame): Dataframe of finance corpus. Each row is a document with attributes: "id", "text".
        faq_corpus (pd.DataFrame): Dataframe of faq corpus. Each row is a document with attributes: "id", "text".
        tokenizer (str): Tokenizer for BM25 to use. Default is "jieba".
        chunking (dict): a dictionary containing the chunking configuration, should have the following keys\n
            "activate": bool, whether to chunk the text\n
            "chunk_size": dict, containing the chunk size for each category\n
            "overlap_size": dict, containing the overlap size for each category
        top_k (dict): Number of documents to retrieve for each query. Default is 5.

        Returns:
        tuple (pd.DataFrame, pd.DataFrame):
            1. Dataframe of retrieved documents. Each row is the top-1 document for each query with attributes: "id", "text", "score".\n
            2. Dataframe of retrieved documents by Jina only. Each row is the top-k documents for each query with attributes: "id", "text", "score".
    """
    df_jina_result = jina_retrieve(
        queries,
        insurance_corpus,
        finance_corpus,
        faq_corpus,
        chunking=chunking,
        top_k=top_k["jina"],
    )

    df_BM25_result = _BM25_retrieve_series(
        queries, df_jina_result, tokenizer=tokenizer, top_k=top_k["BM25"]
    )

    return df_BM25_result, df_jina_result

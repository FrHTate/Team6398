# General
import pandas as pd
import os
import json

# Our modules
import Preprocess.corpus_preprocessor as dp
import Model.retrievers as rt

# import evaluation

# 1. Set parameters for data loading and retrieval
# 1-1. Set paths
path_to_main = os.path.dirname(os.path.abspath(__file__))
paths = {
    "insurance": os.path.join(path_to_main, "reference/insurance.json"),
    "finance": os.path.join(path_to_main, "reference/finance.json"),
    "faq": os.path.join(path_to_main, "reference/faq/pid_map_context.json"),
    "query": os.path.join(
        path_to_main, "dataset/preliminary/questions_preliminary.json"
    ),
    "output": os.path.join(path_to_main, "dataset/preliminary/pred_retrieve.json"),
}

# 1-2. Set Hyperparameters
rewrite = {
    "query": True,
    "text": True,
}
# If we don't want to chunk, just set "activate" to False
chunking = {
    "activate": True,
    "chunk_size": {"insurance": 128, "finance": 256, "jina_default": 1024},
    "overlap_size": {"insurance": 32, "finance": 32, "jina_default": 80},
}
# Each value should be either "full" or int
top_k = {"BM25": 5, "jina": 1}

# 2. Load data
# Each item: {"qid": int, "source": list of str, "query": str, "category": str}
queries = dp.load_queries(paths["query"], rewrite=rewrite["query"])

# Attributes: ["id": int, "text": str]
df_insurance = dp.load_corpus_to_df(
    "insurance",
    paths["insurance"],
    chunking=chunking,
    rewrite=rewrite["text"],
)
df_finance = dp.load_corpus_to_df(
    "finance",
    paths["finance"],
    chunking=chunking,
    rewrite=rewrite["text"],
)
df_faq = dp.load_corpus_to_df("faq", paths["faq"], rewrite=rewrite["text"])

# 3. Retreive top-5 documents by BM25, then retrieve top-1 by Jina.
df_result, _ = rt.BM25_first(
    queries=queries,
    insurance_corpus=df_insurance,
    finance_corpus=df_finance,
    faq_corpus=df_faq,
    tokenizer="jieba",
    chunking=chunking,
    top_k=top_k,
)


# 4. Output the result
def save_json(path, data):
    """
    Save a dictionary to a json file.

    Args:
        path (str): The path to save the json file.
        data (dict): The dictionary to save.

    Returns:
        None
    """
    with open(path, "w") as f:
        # Convert DataFrame to dictionary before saving
        json.dump(data, f, ensure_ascii=False, indent=4)


def answer_to_json(df, path="predicate.json"):
    """
    Find the top-1 document for each query and save the result to a json file.

    Args:
        df (pd.DataFrame): The DataFrame that contains the top-k documents for each query.
        path (str, optional): The path to save the result. Defaults to "predicate.json".

    Returns:
        None
    """
    answer_dict = {}
    answer = []
    for index in df.index:
        answer.append(
            {
                "qid": index + 1,
                "retrieve": int(df.iloc[index]["id"][0]),
            }
        )
    answer_dict["answers"] = answer
    save_json(path, answer_dict)


answer_to_json(df_result, paths["output"])

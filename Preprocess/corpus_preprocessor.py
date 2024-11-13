import json
import pandas as pd
import re


def load_corpus_to_df(
    category,
    source_path,
    chunking={"activate": False, "chunk_size": None, "overlap_size": None},
    summary=False,
    rewrite=False,
):
    """
    Load the corpus from the source path and return a DataFrame with columns "id" and "text".

    Args:
        category (str): the category of the corpus, should be one of "faq", "insurance", "finance"
        source_path (str): the path to the source file of the corpus
        chunking (dict): a dictionary containing the chunking configuration, should have the following keys\n
            "activate": bool, whether to chunk the text\n
            "chunk_size": dict, containing the chunk size for each category\n
            "overlap_size": dict, containing the overlap size for each category

    Returns:
        pd.DataFrame: a DataFrame containing the corpus with columns "id" and "text"
    """
    # attributes in df = ["id", "text"]
    id_list = []
    text_list = []  # if chunking, text == segment; else, text == full text
    with open(source_path, "r") as f:
        data = json.load(f)

    if category == "faq":
        for id, qa_list in data.items():
            for qa in qa_list:
                for answer in qa["answers"]:
                    id_list.append(int(id))
                    text_list.append(qa["question"] + answer)

    elif category == "insurance":
        if chunking["activate"]:
            chunk_size = chunking["chunk_size"][category]
            overlap_size = chunking["overlap_size"][category]
            for doc in data[category]:
                if summary:
                    id_list.append(int(doc["index"]))
                    text_list.append(doc["summary"])
                for i in range(
                    0,
                    len(doc["text"]) - chunk_size + 1,
                    chunk_size - overlap_size,
                ):
                    id_list.append(int(doc["index"]))
                    text_list.append(
                        doc["text"][i : i + chunk_size]
                        + " [標題] "
                        + doc["label"]
                        + " [/標題]."
                    )

        else:
            for doc in data[category]:
                id_list.append(int(doc["index"]))
                text_list.append(doc["text"])

    elif category == "finance":
        if chunking["activate"]:
            chunk_size = chunking["chunk_size"][category]
            overlap_size = chunking["overlap_size"][category]
            for doc in data[category]:
                if summary:
                    id_list.append(int(doc["index"]))
                    text_list.append(doc["summary"])
                for i in range(
                    0,
                    len(doc["text"]) - chunk_size + 1,
                    chunk_size - overlap_size,
                ):
                    id_list.append(int(doc["index"]))
                    if doc["label"] == "":
                        text_list.append(doc["text"][i : i + chunk_size])
                    else:
                        text_list.append(
                            "[標題] "
                            + doc["label"]
                            + " [/標題]. "
                            + doc["text"][i : i + chunk_size]
                        )

        else:
            for doc in data[category]:
                id_list.append(int(doc["index"]))
                text_list.append(doc["text"])

    else:
        raise ValueError(
            "Invalid category, please choose from 'faq', 'insurance', 'finance'"
        )

    if rewrite:
        text_list = [passage_rewrite(text) for text in text_list]

    return pd.DataFrame({"id": id_list, "text": text_list})


def load_queries(source_path, rewrite=False):
    """
    Load the queries from the source path and return a list of queries.

    Args:
        source_path (str): the path to the source file of the queries
        rewrite (bool): whether to rewrite the queries
    
    Returns:
        list: a list of dictionaries with the following attributes\n
            "qid": int, the id of the query\n
            "source": list of int, the sources to answer the query\n
            "query": str, the text of the query\n
            "category": str, the category of the query
    """
    with open(source_path, "r") as f:
        data = json.load(f)

    if rewrite:
        for query in data["questions"]:
            query["query"] = query_rewrite(query["query"])
    return data["questions"]


# Use for preparing stage, contest stage won't provide ground truth
def load_ground_truths(source_path):
    """
    Load the ground truths from the source path and return a list of ground truths.

    Args:
        source_path (str): the path to the source file of the ground truths

    Returns:
        list: a list of dictionaries with the following attributes\n
            "qid": int, the id of the query\n
            "retrieve": int, the retrieved document id\n
            "category": str, the category of the query
    """
    with open(source_path, "r") as f:
        data = json.load(f)
    return data["ground_truths"]


def chinese_to_arabic(chinese_numeral):
    """
    Convert Chinese numerals to Arabic numerals.

    Args:
        chinese_numeral (str): the Chinese numeral to convert
    
    Returns:
        int: the coreesponding Arabic numeral
    """
    chinese_numeral_map = {
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    result = 0

    if "十" in chinese_numeral:
        parts = chinese_numeral.split("十")
        if parts[0] == "":
            result += 10
        else:
            result += chinese_numeral_map[parts[0]] * 10
        if len(parts) > 1 and parts[1] != "":
            result += chinese_numeral_map[parts[1]]
    else:
        for char in chinese_numeral:
            result = result * 10 + chinese_numeral_map[char]

    return result


def convert_text_dates(text):
    """
    Convert Chinese representation of dates to Arabic numerals.

    Args:
        text (str): the text to convert

    Returns:
        str: the text with Chinese representation of dates converted to Arabic numerals
    """
    # Convert 民國 (ROC year) to Gregorian year only for three-digit years,
    # ensuring not to convert already existing four-digit Gregorian years.
    text = re.sub(r"(?<!\d)(\d{3})年", lambda m: f"{int(m.group(1)) + 1911}年", text)

    # Convert fully Chinese representation of dates to Arabic numerals
    text = re.sub(
        r"([〇一二三四五六七八九十]+)年([〇一二三四五六七八九十]+)月([〇一二三四五六七八九十]+)日",
        lambda m: f"{chinese_to_arabic(m.group(1)) + 1911}年{chinese_to_arabic(m.group(2))}月{chinese_to_arabic(m.group(3))}日",
        text,
    )

    text = re.sub(
        r"([〇一二三四五六七八九十]+)年([〇一二三四五六七八九十]+)月",
        lambda m: f"{chinese_to_arabic(m.group(1)) + 1911}年{chinese_to_arabic(m.group(2))}月",
        text,
    )

    text = re.sub(
        r"([一二三四五六七八九十]+)月([〇一二三四五六七八九十]+)日",
        lambda m: f"{chinese_to_arabic(m.group(1))}月{chinese_to_arabic(m.group(2))}日",
        text,
    )

    return text


def query_rewrite(query):
    """
    Rewrite the query to standardize the format, such as converting Chinese numerals to Arabic numerals and expanding company abbreviations.

    Args:
        query (str): the query to rewrite
    
    Returns:
        str: the rewritten query
    """
    # years = re.findall(r"(\d{4})年", query)
    # if years:
    #     years = years[0]
    # else:
    #     years = ""
    n = [
        ("1", "一", f"Q1"),
        ("2", "二", f"Q2"),
        ("3", "三", f"Q3"),
        ("4", "四", f"Q4"),
    ]
    query_rewrite = query
    for season in n:
        if f"第{season[0]}季" in query or f"第{season[1]}季" in query:
            query_rewrite = query.replace(f"第{season[0]}季", season[2]).replace(
                f"第{season[1]}季", season[2]
            )

    query_rewrite = convert_text_dates(query_rewrite)

    company_names = {
        # "聯發科": "聯發科技股份有限公司",
        "台化": "台灣化學纖維股份有限公司",
        # "台達電": "台達電子工業股份有限公司",
        "台泥": "台灣水泥股份有限公司",
        # "華碩": "華碩電腦股份有限公司",
        # "瑞昱": "瑞昱半導體股份有限公司",
        # "長榮": "長榮海運股份有限公司",
        "聯電": "聯華電子股份有限公司",
        # "智邦": "智邦科技股份有限公司",
        # "和泰汽車": "和泰汽車股份有限公司",
        "中鋼": "中國鋼鐵股份有限公司",
        # "鴻海": "鴻海精密工業股份有限公司",
        # "亞德客": "亞德客國際集團及其子公司",
        # "統一企業": "統一企業股份有限公司",
        # "國巨": "國巨股份有限公司",
        # "研華": "研華股份有限公司",
        # "中華電信": "中華電信股份有限公司",
        # "光寶": "光寶科技股份有限公司",
        # "台積電": "台灣積體電路製造股份有限公司",
        # "台永電": "台灣永電股份有限公司",
        # "合作金庫": "合作金庫商業銀行股份有限公司",
    }

    for abbr, full_name in company_names.items():
        if abbr in query_rewrite and full_name not in query_rewrite:
            query_rewrite = query_rewrite.replace(abbr, f"{abbr}({full_name})")

    return query_rewrite


def passage_rewrite(passage):
    """
    Rewrite the passage to standardize the format (converting Chinese numerals to Arabic numerals).

    Args:
        passage (str): the passage to rewrite
    
    Returns:
        str: the rewritten passage
    """
    n = [
        ("1", "一", f"Q1"),
        ("2", "二", f"Q2"),
        ("3", "三", f"Q3"),
        ("4", "四", f"Q4"),
    ]
    passage_rewrite = passage
    for season in n:
        if f"第{season[0]}季" in passage or f"第{season[1]}季" in passage:
            passage_rewrite = passage_rewrite.replace(
                f"第{season[0]}季", season[2]
            ).replace(f"第{season[1]}季", season[2])

    # Fix the re.sub() calls by correctly using backreferences without additional quotes
    passage_rewrite = re.sub(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日", r"\1/\2/\3", passage_rewrite
    )
    passage_rewrite = re.sub(r"(\d{4})年(\d{1,2})月", r"\1/\2", passage_rewrite)
    passage_rewrite = re.sub(r"(\d{1,2})月(\d{1,2})日", r"\1/\2", passage_rewrite)

    return passage_rewrite

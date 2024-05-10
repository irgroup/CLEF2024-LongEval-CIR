import pandas as pd
import pyterrier as pt
import yaml
import os
from src.load_index import load_index, load_topics, load_qrels, tag
from src.extend_runs import extend_run_full
import sqlite3
from repro_eval.Evaluator import RpdEvaluator
import pytrec_eval
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from argparse import ArgumentParser
from repro_eval.util import arp, arp_scores

if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

with open("data/LongEval/metadata.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

results_path = "data/results/relevance_feedback/"
metadata_path = "data/results/metadata/"


def extract_top_terms(texts, top_n=10, query=None):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0)
    data = []
    for col, term in enumerate(feature_names):
        data.append((term, sums[0, col]))
    top_terms = sorted(data, key=lambda x: x[1], reverse=True)[:top_n]
    return top_terms



def get_relevance_feedback_topics(topics, history, conn):
    # get query_map
    query = """SELECT topic.queryid as qid_1, T2.queryid as qid_2 from topic
    JOIN topic as T2
    ON topic.text_fr = T2.text_fr
    WHERE T2.sub_collection IN (%s)"""% ",".join("?" * len(history))

    query_map = pd.read_sql_query(query, conn, params=history)

    new_topics = []
    extended_topics = []
    for _, topic in topics.iterrows():
        new_topic = {"qid": topic["qid"], "query": topic["query"], "query_original": topic["query"]}
        
        # get queryies for potential extension
        queryies_to_extend = query_map[query_map["qid_1"] == topic["qid"]]["qid_2"].tolist()
        
        if len(queryies_to_extend) == 0:
            # extended_topics.append(new_topic)
            new_topics.append(topic["qid"])
            print("No similar topics found", topic["qid"], topic["query"])
            continue

        # get rel docs
        query = """SELECT url, text_en
        FROM qrel
        JOIN document ON qrel.docid = document.docid
        WHERE queryid IN (%s)
        AND relevance > 0"""% ",".join("?" * len(queryies_to_extend))
        
        rel_docs = pd.read_sql_query(query, conn, params=queryies_to_extend)
        texts = rel_docs.drop_duplicates(subset="url")["text_en"].str.replace("\n", " ").tolist()
        if len(texts) == 0:
            # extended_topics.append(new_topic)
            new_topics.append(topic["qid"])
            print("No relevant docs found", topic["qid"], topic["query"])
            continue
        extension_terms = [item[0] for item in extract_top_terms(texts)]
        extension_terms = " ".join(extension_terms)
        new_topic["query"] = new_topic["query"] + " " + extension_terms
        
        extended_topics.append(new_topic)
    
    return new_topics, extended_topics
        

def main(args):
    split_name = "train" if args.train else "test"
    topics_name = args.topics if args.topics else args.index
    
    history = args.history
    assert topics_name not in history, "The topics should not be in the history."    
    print(">>> Use index:", args.index)
    print(">>> Use topic set:", topics_name)
    print(">>> Use history:", history)
    
    index = load_index(args.index)
    topics = load_topics(topics_name, split_name)
          
    conn = sqlite3.connect("data/database.db")
    
    new_topics, extended_topics = get_relevance_feedback_topics(topics, history, conn)
    
    new_topics = topics[topics["qid"].isin(new_topics)]
    extended_topics = pd.DataFrame(extended_topics)
  


    BM25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True)
    rm3_pipe = BM25 >> pt.rewrite.RM3(index) >> BM25

    print(">>> Run BM25 with relevance feedback")
    run_with_feedback = BM25.transform(extended_topics)
    
    print(">>> Run RM3 with pseudo relevance feedback")
    run_with_pseudo_feedback = rm3_pipe.transform(new_topics)
    
    merged_run = pd.concat([run_with_feedback, run_with_pseudo_feedback])
    
    pt.io.write_results(merged_run, results_path + f"/CIR_BM25_D-{args.index}_T-{topics_name}_rf{"".join(history)}")
    print(">>> Extended run saved")

if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file (WT, ST or LT)",
    )
    parser.add_argument(
        "--topics",
        required=False,
        type=str,
        help="topics of the subcollection.",
    )
    parser.add_argument(
        "--train",
        required=False,
        action="store_true",
        help="Use the train topics to create the.",
    )
    parser.add_argument(
        "--history",
        nargs="+",
        required=True,
        help="History to be used.",
    )
    
    args = parser.parse_args()
    main(args)

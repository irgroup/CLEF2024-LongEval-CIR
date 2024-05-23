#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BM25 system that is reranked by boosting documents based on known qrels.

Example:
    Create runs on the train topics of the given index::

        $ python -m systems.BM25+qrel_boost --index t3 --history t1 t2 --l 0.5 --m 2
"""
from argparse import ArgumentParser

import pyterrier as pt  # type: ignore

from src.load_index import load_index, load_topics, load_qrels, tag
import yaml
import pandas as pd
import pyterrier as pt
import yaml
from src.load_index import load_index, load_topics, load_qrels, tag
from src.extend_runs import extend_run_full
import sqlite3


if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

with open("data/LongEval/metadata.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

results_path = "data/results/trec/"
metadata_path = "data/results/metadata/"


def extend_with_qrels(run_name, history):
    conn = sqlite3.connect("data/database.db")
    query = "SELECT * FROM qrel"
    qrels = pd.read_sql_query(query, conn)

    run = pd.read_csv(results_path + run_name, sep=" ")
    qrels_map = qrels[["key", "relevance"]].set_index("key").to_dict()["relevance"]

    def get_qrel(row, subcollection):
        query_id = row[f"queryid_{subcollection}"]
        doc_id = row[f"docid_{subcollection}"]
        if isinstance(query_id, str) and isinstance(doc_id, str):
            return qrels_map.get(
                row[f"queryid_{subcollection}"] + row[f"docid_{subcollection}"], None
            )
        else:
            return None

    for subcollection in history:
        run[f"qrel_{subcollection}"] = run.apply(
            get_qrel, subcollection=subcollection, axis=1
        )

    return run


def qrel_boost(run, history, _lambda=0.5, mu=2):
    # min max normalization per topic
    run["score"] = run.groupby("queryid")["score"].transform(lambda x: x / x.max())

    for subcollection in history:
        # Relevant
        run.loc[run[f"qrel_{subcollection}"] == 1, "score"] = (
            run.loc[run[f"qrel_{subcollection}"] > 0, "score"] * _lambda**2
        )
        run.loc[run[f"qrel_{subcollection}"] > 0, "score"] = (
            run.loc[run[f"qrel_{subcollection}"] > 0, "score"] * (_lambda**2) * mu
        )

        # All Not Relevant
        run.loc[
            (run[f"qrel_{subcollection}"] == 0) | (run[f"qrel_{subcollection}"].isna()),
            "score",
        ] = (
            run.loc[
                (run[f"qrel_{subcollection}"] == 0)
                | (run[f"qrel_{subcollection}"].isna()),
                "score",
            ]
            * (1 - _lambda) ** 2
        )

    run = (
        run.sort_values(["queryid", "score"], ascending=False)
        .groupby("queryid")
        .head(1000)
    )
    run["rank"] = run.groupby("queryid")["score"].rank(ascending=False).astype(int)

    run = run[["queryid", "0", "docid", "score", "rank", "run"]].rename(
        columns={"queryid": "qid", "docid": "docno"}
    )

    return run


def main(args):
    split_name = "train" if args.train else "test"
    topics_name = args.topics if args.topics else args.index

    _lambda = args.l
    mu = args.m

    history = args.history
    assert topics_name not in history, "The topics should not be in the history."

    index = load_index(args.index)
    topics = load_topics(topics_name, split_name)

    run_name = f"/CIR_BM25+qrel_boost_D-{args.index}_T-{topics_name}-{''.join(history)}_l-{_lambda}_m-{mu}"

    print(">>> Use history:", history)

    # BM25 top 1500 as baseline
    BM25 = pt.BatchRetrieve(
        index, wmodel="BM25", verbose=True, num_results=1500
    )  # retrieve more results to filter
    pt.io.write_results(BM25(topics), results_path + run_name + "-long")

    extend_run_full(results_path + run_name + "-long")

    history_complete = history + [
        topics_name
    ]  # we need the topic sub-collection to merge the qrels
    run = extend_with_qrels(run_name + "-long", history_complete)

    run = qrel_boost(run, history, _lambda, mu)

    pt.io.write_results(run, results_path + run_name)


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
    parser.add_argument("-l", "--lambda", type=float, default=0.5)
    parser.add_argument("-m", "--mu", type=float, default=2.0)

    args = parser.parse_args()
    main(args)

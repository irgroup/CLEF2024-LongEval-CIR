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


def time_fuse(run_recent, run_old, _lambda=0.5):
    qid_ranking_groups = run_old.groupby("qid")
    qid_ranking_dict = {
        qid: list(ranking["docno"]) for qid, ranking in qid_ranking_groups
    }

    def weigh(row):
        if not qid_ranking_dict.get(row["qid"]):
            # These topics will be boosted down, an empty set is returned
            print("Could not find", row["qid"])

        if row["docno"] in qid_ranking_dict.get(row["qid"], []):
            return row["score"] * _lambda**2
        else:
            return row["score"] * (1 - _lambda) ** 2

    reranking = run_recent.copy()

    # min max normalization per topic
    reranking["score"] = reranking.groupby("qid")["score"].transform(
        lambda x: x / x.max()
    )

    # weight if in old ranking
    reranking["score"] = reranking.progress_apply(weigh, axis=1)
    reranking = (
        reranking.sort_values(["qid", "score"], ascending=False)
        .groupby("qid")
        .head(1000)
    )
    reranking["rank"] = (
        reranking.groupby("qid")["score"].rank(ascending=False).astype(int)
    )
    return reranking


def main(args):
    _lambda = args.l
    run_new_path = args.new
    run_old_path = args.old

    run_new = pt.io.read_results(run_new_path)
    run_old = pt.io.read_results(run_old_path)

    run_reranked = time_fuse(run_new, run_old, _lambda=_lambda)

    pt.io.write_results(run_reranked, run_new_path + f"_ft-{_lambda}", format="trec")


if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")
    parser.add_argument(
        "--new",
        type=str,
        required=True,
        help="Path top the new run.",
    )
    parser.add_argument(
        "--old",
        required=False,
        type=str,
        help="Path to the old run.",
    )
    parser.add_argument("-l", "--lambda", type=float, default=0.5)

    args = parser.parse_args()
    main(args)

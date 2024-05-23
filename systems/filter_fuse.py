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
import numpy as np
from repro_eval.Evaluator import RpdEvaluator
from ranx import Run, fuse
import os

if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

with open("data/LongEval/metadata.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

results_path = "data/results/trec/"
metadata_path = "data/results/metadata/"


def filter_and_fuse(run_recent, old_runs: list):
    qid_ranking_groups = run_recent.groupby("qid")
    qid_ranking_dict_recent = {
        qid: pd.Series(ranking["score"].values, ranking["docno"]).to_dict()
        for qid, ranking in qid_ranking_groups
    }

    runs = [Run.from_dict(qid_ranking_dict_recent)]

    for run_old in old_runs:
        qid_ranking_groups = run_old.groupby("qid")
        qid_ranking_dict_old = {
            qid: pd.Series(ranking["score"].values, ranking["docno"]).to_dict()
            for qid, ranking in qid_ranking_groups
        }
        for qid, ranking in qid_ranking_dict_old.items():
            docs_recent = qid_ranking_dict_recent.get(qid).keys()
            qid_ranking_dict_old[qid] = {
                docid: score for docid, score in ranking.items() if docid in docs_recent
            }
        runs.append(Run.from_dict(qid_ranking_dict_old))

    combined_run = fuse(runs=runs, method="rrf")

    return combined_run


# Load history of runs
def load_history_runs(history, sub_collection):
    history_index = ["D-" + i for i in history]
    old_runs = []
    # TODO: fix hardcoded paths
    base_path = "data"
    runs_path = "results/trec"

    for name in os.listdir(os.path.join(base_path, runs_path)):
        if sub_collection in name and name.endswith("extended"):
            for i in history_index:
                if i in name:
                    run = pt.io.read_results(os.path.join(base_path, runs_path, name))
                    old_runs.append(run)
    return old_runs


# find core qids
def core_topics(run_new, old_runs):
    topic_sets = []
    for i in old_runs:
        topic_sets.append(set(i["qid"]))
    topic_sets.append(set(run_new["qid"]))

    core_topics = set.intersection(*topic_sets)
    print(
        "Found known documents for:",
        len(core_topics),
        "of",
        len(run_new["qid"].unique()),
        "topics",
    )
    return core_topics


def clean_runs(run_new, old_runs, core_topics):
    old_runs_cleaned = []
    for run in old_runs:
        old_runs_cleaned.append(run[run["qid"].isin(core_topics)])
    old_runs = old_runs_cleaned

    run_new_cleaned = run_new[run_new["qid"].isin(core_topics)]

    return run_new_cleaned, old_runs


def main(args):

    history = args.history

    run_new = pt.io.read_results(args.new)

    old_runs = load_history_runs(history, args.index)

    core_topics = core_topics(run_new, old_runs)
    missing_topics = set(run_new["qid"]) - set(core_topics)

    run_new, runs_old = clean_runs(run_new, old_runs, core_topics)

    run_reranked = filter_and_fuse(run_new, runs_old)

    run_path = args.new + f'_rr-ff{"".join(history)}'
    missing_topic_ranking = run_new[run_new["qid"].isin(missing_topics)]
    run_reranked.save(run_path, kind="trec")

    reranked_run = pd.read_csv(
        run_path,
        sep=" ",
        names=["qid", "Q0", "docno", "rank", "score", "name"],
    )

    missing_topic_ranking["Q0"] = "Q0"
    pd.concat([reranked_run, missing_topic_ranking], ignore_index=True).to_csv(
        run_path, sep=" ", header=None, index=None
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")
    parser.add_argument(
        "--new",
        type=str,
        required=True,
        help="Path top the new run.",
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Name of the dataset in the config file (e.g. t0)",
    )
    parser.add_argument("-l", "--lambda", type=float, default=0.5)
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MonoT5 baseline system for LongEval 2023.

This system uses the monoT5 reranker and a BM25 first stage ranker.

Example:
    Run the system with the following command::

        $ python -m systems.monoT5 --index WT
        $ python -m systems.monoT5 --index WT
        $ python -m systems.monoT5 --index WT --model monoT5-MS-WT
        $ python -m systems.monoT5 --index WT --model monoT5-MS-WT-train
        $ python -m systems.monoT5 --index WT --model monoT5-WT
        $ python -m systems.monoT5 --index WT --model monoT5-WT-train
"""
from argparse import ArgumentParser

import pyterrier as pt  # type: ignore
from pyterrier_t5 import MonoT5ReRanker
import yaml  # type: ignore

from src.load_index import load_index, load_topics, load_qrels, tag

# logger.setLevel("INFO")

with open("data/LongEval/metadata.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

results_path = "data/results/trec/"
metadata_path = "data/results/metadata/"


def get_system(index: pt.IndexFactory, model_path: str = "") -> pt.BatchRetrieve:
    if model_path:
        model_path = "data/models/" + model_path
        monoT5 = MonoT5ReRanker(verbose=True, batch_size=8, model=model_path)
    else:
        monoT5 = MonoT5ReRanker(verbose=True, batch_size=10)

    bm25 = pt.BatchRetrieve(
        index, wmodel="BM25", metadata=["docno", "text"], verbose=True
    )

    mono_pipeline = bm25 >> pt.text.get_text(index, "text", by_query=True) >> monoT5

    return mono_pipeline


def main(args):
    # name = "BM25+" + args.model if args.model else "monoT5"
    # run_tag = tag(name, args.index)

    split_name = "train" if args.train else "test"
    topics_name = args.topics if args.topics else args.index

    print(">>> Use index:", args.index)
    print(">>> Use topic set:", topics_name)
    print(">>> Use topic split:", split_name)

    index = load_index(args.index)
    topics = load_topics(topics_name, split_name)

    system = get_system(index)

    pt.io.write_results(
        system(topics),
        results_path + f"/CIR_BM25+monoT5_D-{args.index}_T-{topics_name}",
    )
    # write_metadata_yaml(
    #     config["metadata_path"] + run_tag + ".yml",
    #     {
    #         "tag": run_tag,
    #         "method": {
    #             "retrieval": {
    #                 "1": {
    #                     "name": "bm25",
    #                     "method": "org.terrier.matching.models.BM25",
    #                     "k_1": "1.2",
    #                     "k_3": "8",
    #                     "b": "0.75",
    #                 },
    #                 "2": {
    #                     "name": "monoT5 reranker",
    #                     "method": "pyterrier_t5",
    #                     "model": args.model if args.model else "monoT5",
    #                     "passages": False,
    #                 },
    #             },
    #         },
    #     },
    # )


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

    main(parser.parse_args())

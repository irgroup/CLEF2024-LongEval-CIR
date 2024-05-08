import os
from typing import Tuple

import pandas as pd  # type: ignore
import numpy as np

import pyterrier as pt  # type: ignore
import yaml  # type: ignore

if not pt.started():
    pt.init()

with open("data/LongEval/metadata.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def load_index(index_name: str) -> pt.IndexFactory:
    """Load an index from disk.

    Args:
        index_name (str): Name of the index as specified in the config file.

    Returns:
        pt.IndexFactory: The loaded index.
    """
    index = pt.IndexFactory.of(os.path.join(BASE_PATH, "index", index_name))
    print(
        ">>> Loaded index with ",
        index.getCollectionStatistics().getNumberOfDocuments(),
        "documents.",
    )
    return index


def load_topics(subcollection: str, split: str) -> pd.DataFrame:
    """Load the topics for a dataset.

    Args:
        index_name (str): Name of the dataset split as specified in the config file.
        split (str): The split to load, either "train" or "test".

    Returns:
        pd.DataFrame: The topics.
    """

    return pt.io.read_topics(
        os.path.join(
            BASE_PATH,
            config["subcollections"][subcollection]["topics"][split]["trec"]["en"],
        )
    )


def load_qrels(subcollection: str, split: str) -> pd.DataFrame:
    """Load the qrels for a dataset.

    Args:
        index_name (str): Name of the dataset split as specified in the config file.
        split (str): The split to load, either "train" or "test".

    Returns:
        pd.DataFrame: The qrels.
    """
    if config["subcollections"][subcollection]["qrels"][split]:
        return pt.io.read_qrels(
            os.path.join(
                BASE_PATH,
                config["subcollections"][subcollection]["qrels"][split],
            )
        )
    else:
        return pd.DataFrame()


# def setup_system(
#     index_name: str, train: bool = True
# ) -> Tuple[pt.IndexFactory, pd.DataFrame, pd.DataFrame]:
#     """Load the index, topics and qrels for a dataset that is allready indexed.

#     Args:
#         index_name (str): Name of the dataset split as specified in the config file.
#         train (bool, optional): Return the train or the test split. Defaults to True.

#     Returns:
#         (pt.IndexFactory, pd.DataFrame, pd.DataFrame): The index, topics and qrels.
#     """
#     split = "train" if train else "test"

#     index = _load_index(index_name)

#     topics = pt.io.read_topics(
#         os.path.join(
#             BASE_PATH,
#             config["subcollections"][index_name]["topics"][split]["trec"]["en"],
#         )
#     )
#     if config["subcollections"][index_name]["qrels"][split]:
#         qrels = pt.io.read_qrels(
#             os.path.join(
#                 BASE_PATH,
#                 config["subcollections"][index_name]["qrels"][split],
#             )
#         )
#     else:
#         qrels = pd.DataFrame()

#     return index, topics, qrels


def tag(system: str, index: str) -> str:
    """Create a tag for the run."""
    return f"CIR_{system}.{index}"


def get_train_splits(topics, qrels):
    def filter_ids(topics):
        needed_ids = list(topics["qid"].unique())  # needed ids
        qrels_split = qrels[qrels["qid"].isin(needed_ids)]
        diff = len(needed_ids) - len(qrels_split["qid"].unique())
        return qrels_split

    # split topics
    train_topics, validation_topics, test_topics = np.split(
        topics, [int(0.6 * len(topics)), int(0.8 * len(topics))]
    )

    # split qrels
    train_qrels = filter_ids(train_topics)
    validation_qrels = filter_ids(validation_topics)
    test_qrels = filter_ids(test_topics)

    return (
        train_topics,
        validation_topics,
        test_topics,
        train_qrels,
        validation_qrels,
        test_qrels,
    )

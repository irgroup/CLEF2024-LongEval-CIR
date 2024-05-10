import pandas as pd
import sqlite3
from argparse import ArgumentParser
from tqdm import tqdm


def extend_run_full(run_path):
    # Connect to the SQLite database
    conn = sqlite3.connect("data/database.db")

    for i in run_path.split("_"):
        if i.startswith("D-"):
            index_name = i[-2:]

    print(">>> Loaded run")
    run = pd.read_csv(
        run_path,
        sep=" ",
        names=["queryid", "0", "docid", "relevance", "score", "run"],
        index_col=False,
    )

    # Load doc map
    print(">>> Load doc map")
    docids = run["docid"].unique()

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    query_base = "SELECT docid, url FROM Document WHERE docid IN ({})"

    results = []
    for chunk in chunker(docids, 100_000):
        placeholders = ", ".join(["?"] * len(chunk))
        query = query_base.format(placeholders)
        result = pd.read_sql_query(query, conn, params=chunk)
        results.append(result)

    docmapper = pd.concat(results, ignore_index=True)

    query_base = "SELECT docid, url, sub_collection FROM Document WHERE url IN ({})"

    results = []
    for chunk in tqdm(
        chunker(docmapper["url"].unique(), 10_000), total=len(docmapper) / 10_000
    ):
        placeholders = ", ".join(["?"] * len(chunk))
        query = query_base.format(placeholders)
        params = list(chunk)
        result = pd.read_sql_query(query, conn, params=params)
        results.append(result)

    docid_map = pd.concat(results, ignore_index=True)
    docid_map = docid_map.pivot(index="url", columns="sub_collection", values="docid")

    # Querymap
    print(">>> Load query map")
    queryids = run["queryid"].unique()
    query = "SELECT queryid, text_fr FROM Topic WHERE queryid IN (%s);" % ",".join(
        "?" * len(queryids)
    )
    querymapper = pd.read_sql_query(query, conn, params=queryids)

    query = (
        "SELECT queryid, text_fr, text_en, sub_collection FROM Topic WHERE text_fr IN (%s);"
        % ",".join("?" * len(querymapper["text_fr"].unique()))
    )
    query_text_fr_map = pd.read_sql_query(
        query, conn, params=querymapper["text_fr"].unique()
    )
    query_text_fr_map = query_text_fr_map.pivot(
        index=["text_fr", "text_en"], columns="sub_collection", values="queryid"
    )

    # Merge
    print(">>> Extend run")
    run_docids_extended = run.merge(
        docid_map.add_prefix("docid_"),
        left_on="docid",
        right_on=f"docid_{index_name}",
        how="left",
    )
    run_extended = run_docids_extended.merge(
        query_text_fr_map.add_prefix("queryid_"),
        left_on="queryid",
        right_on=f"queryid_{index_name}",
        how="left",
    )

    run_extended.to_csv(run_path[:-3] + "_extended." + index_name, sep=" ", index=False)


def extend_topics(run_path):
    print("not implemented")
    print(">>> Extend topics", run_path)


def extend_documents(run_path):
    # Connect to the SQLite database
    conn = sqlite3.connect("data/database.db")

    # subcollection = run_path.split(".")[-1]
    index = run_path.split("_")[-2][-2:]
    topics = run_path.split("_")[-1][-2:]
    print(">>> Loaded run")

    run = pd.read_csv(
        run_path,
        sep=" ",
        names=["queryid", "0", "docid", "relevance", "score", "run"],
        index_col=False,
    )

    topic_set = set(run["queryid"])

    # Load doc map
    print(">>> Load doc map")
    docids = run["docid"].unique()

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    query_base = "SELECT docid, url FROM Document WHERE docid IN ({})"

    results = []
    for chunk in chunker(docids, 100_000):
        placeholders = ", ".join(["?"] * len(chunk))
        query = query_base.format(placeholders)
        result = pd.read_sql_query(query, conn, params=chunk)
        results.append(result)

    doc_url = pd.concat(results, ignore_index=True)

    query_base = (
        "SELECT docid, url FROM Document WHERE sub_collection = ? AND url IN ({})"
    )

    results = []
    for chunk in tqdm(
        chunker(doc_url["url"].unique(), 100_000), total=len(doc_url) / 100_000
    ):
        placeholders = ", ".join(["?"] * len(chunk))
        query = query_base.format(placeholders)
        params = [topics] + list(chunk)
        result = pd.read_sql_query(query, conn, params=params)
        results.append(result)

    docid_map = pd.concat(results, ignore_index=True)

    # merge to create map of docids
    docid_map = doc_url.merge(
        docid_map.add_prefix("new_"), left_on="url", right_on="new_url", how="left"
    )[["docid", "new_docid"]]

    # Merge run with doc map
    run = run.merge(docid_map, left_on="docid", right_on="docid", how="left")

    run = run[["queryid", "0", "new_docid", "relevance", "score", "run"]].rename(
        columns={"new_docid": "docid"}
    )

    run = run.dropna()

    topic_set_new = set(run["queryid"])
    if set.difference(topic_set, topic_set_new):
        print("Lost queries by deleting docs:")
        print(set.difference(topic_set, topic_set_new))

    run.to_csv(run_path + "_extended", sep=" ", index=False, header=False)


def main():
    parser = ArgumentParser(description="Load the dataset to a database.")

    # Create a subparser object, make sure it's required with Python 3.7+
    subparsers = parser.add_subparsers(help="sub-command help")
    subparsers.required = True

    topics_extender = subparsers.add_parser(
        "topics", help="Extend topic IDs with topic ids of other subcollections."
    )
    topics_extender.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to the run",
    )
    topics_extender.set_defaults(func=extend_topics)

    document_extender = subparsers.add_parser(
        "documents",
        help="Extend document IDs with document ids of other subcollections.",
    )
    document_extender.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to the run",
    )
    document_extender.set_defaults(func=extend_documents)

    document_extender = subparsers.add_parser(
        "full",
        help="Extend document IDs with document ids of other subcollections.",
    )
    document_extender.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to the run",
    )
    document_extender.set_defaults(func=extend_run_full)

    args = parser.parse_args()

    # Execute the function associated with the chosen subcommand
    if hasattr(args, "func"):
        args.func(args.run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

import pandas as pd
import sqlite3
from argparse import ArgumentParser


def extend_run(run_path):
    # Connect to the SQLite database
    conn = sqlite3.connect("data/database.db")

    subcollection = run_path.split(".")[-1]
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
    query = "SELECT docid, url FROM Document WHERE docid IN (%s);" % ",".join(
        "?" * len(docids)
    )
    docmapper = pd.read_sql_query(query, conn, params=docids)

    query = (
        "SELECT docid, url, sub_collection FROM Document WHERE url IN (%s);"
        % ",".join("?" * len(docmapper["url"].unique()))
    )
    docmap = pd.read_sql_query(query, conn, params=docmapper["url"].unique())

    docmap = docmap.pivot(index="url", columns="sub_collection", values="docid")

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
        docmap.add_prefix("docid_"),
        left_on="docid",
        right_on=f"docid_{subcollection}",
        how="left",
    )
    run_extended = run_docids_extended.merge(
        query_text_fr_map.add_prefix("queryid_"),
        left_on="queryid",
        right_on=f"queryid_{subcollection}",
        how="left",
    )

    run_extended.to_csv(
        run_path[:-3] + "_extended." + subcollection, sep=" ", index=False
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Create an pyterrier index from a config.")

    # input arguments
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to the run",
    )
    args = parser.parse_args()

    extend_run(args.run)

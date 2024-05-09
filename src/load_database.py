from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import os
import tqdm
import json
import yaml
import pandas as pd
from argparse import ArgumentParser

BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

with open("data/LongEval/metadata.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "Document"
    docid: Mapped[str] = mapped_column(primary_key=True)
    text_en: Mapped[str] = mapped_column(String(), nullable=False)
    text_fr: Mapped[str] = mapped_column(String(), nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(), unique=False)
    sub_collection: Mapped[str] = mapped_column(String(), nullable=False)

    def __repr__(self) -> str:
        return f"Document(docid={self.docid!r}, sub_collection={self.sub_collection!r}, text={self.text_en!r})"


class Topic(Base):
    __tablename__ = "Topic"
    queryid: Mapped[str] = mapped_column(primary_key=True)
    text_en: Mapped[str] = mapped_column(String(), nullable=False)
    text_fr: Mapped[Optional[str]] = mapped_column(String(), nullable=True)
    sub_collection: Mapped[str] = mapped_column(String(), nullable=False)
    split: Mapped[str] = mapped_column(String(), nullable=False)

    def __repr__(self) -> str:
        return f"Query(queryid={self.queryid!r}, sub_collection={self.sub_collection!r}, text={self.text_en!r})"


class Qrel(Base):
    __tablename__ = "Qrel"
    queryid: Mapped[str] = mapped_column(ForeignKey("Topic.queryid"), primary_key=True)
    docid: Mapped[str] = mapped_column(ForeignKey("Document.docid"), primary_key=True)
    relevance: Mapped[int] = mapped_column()
    sub_collection: Mapped[str] = mapped_column(String(), nullable=False)
    split: Mapped[str] = mapped_column(String(), nullable=False)

    def __repr__(self) -> str:
        return f"Qrel(queryid={self.queryid!r}, docid={self.docid!r}, relevance={self.relevance!r})"


def seed_database():
    print(BASE_PATH)
    engine = create_engine("sqlite:///data/database.db", echo=False)

    Base.metadata.create_all(engine)


def document_generator():
    for subcollection in config["subcollections"]:
        print(">>> Importing sub-collection:", subcollection)

        # load urls
        urls_path = os.path.join(
            BASE_PATH, config["subcollections"][subcollection]["metadata"]
        )
        print(urls_path)
        with open(urls_path, "r") as f:
            urls = {}
            urls_file = f.readlines()
            for line in urls_file:
                docid, url = line.strip("\n").split("\t")
                urls[docid] = url

        print(len(urls.keys()))

        # docs
        for language in config["subcollections"][subcollection]["documents"]["json"]:
            if language == "fr":
                continue
            docs_path = os.path.join(
                BASE_PATH,
                config["subcollections"][subcollection]["documents"]["json"][language],
            )

            for split in os.listdir(os.path.join(BASE_PATH, docs_path)):
                split_path = os.path.join(docs_path, split)

                # open docs
                with open(os.path.join(BASE_PATH, split_path), "r") as f:
                    documents = json.load(f)

                for document in documents:
                    doc = Document(
                        docid=document["id"],
                        text_en=document["contents"],
                        url=urls.get(document["id"]),
                        sub_collection=subcollection,
                    )

                    yield doc


def topic_generator():
    for subcollection in config["subcollections"]:
        print(">>> Importing sub-collection:", subcollection)
        topics = config["subcollections"][subcollection]["topics"]

        for split in topics.keys():
            split_topics = topics[split]["tsv"]
            en = split_topics["en"]
            fr = split_topics["fr"]
            fr = pd.read_csv(
                os.path.join(BASE_PATH, fr), sep="\t", names=["queryid", "text"]
            )
            en = pd.read_csv(
                os.path.join(BASE_PATH, en), sep="\t", names=["queryid", "text"]
            )

            df = fr.merge(en, on="queryid", suffixes=("_fr", "_en"))

            for _, row in df.iterrows():
                topic = Topic(
                    queryid=row["queryid"],
                    text_en=row["text_en"],
                    text_fr=row["text_fr"],
                    sub_collection=subcollection,
                    split=split,
                )
                yield topic


def qrel_generator():
    for subcollection in config["subcollections"]:
        print(">>> Importing sub-collection:", subcollection)
        qrel_sets = config["subcollections"][subcollection]["qrels"]

        for split in qrel_sets.keys():
            qrels = qrel_sets[split]

            if qrels:
                df = pd.read_csv(
                    os.path.join(BASE_PATH, qrels),
                    sep=" ",
                    names=["queryid", "0", "docid", "relevance"],
                )
                for _, row in df.iterrows():
                    qrel = Qrel(
                        queryid=row["queryid"],
                        docid=row["docid"],
                        relevance=row["relevance"],
                        sub_collection=subcollection,
                        split=split,
                    )
                    yield qrel


def batch_import(generator):
    c = 0
    engine = create_engine("sqlite:///data/database.db", echo=False)
    with Session(engine) as session:
        for doc in tqdm.tqdm(generator()):
            session.add(doc)
            if c % 150000 == 0:
                session.commit()
            c += 1

        session.commit()


def load_documents():
    print(">>> Importing documents")
    batch_import(document_generator)


def load_topics():
    print(">>> Importing topics")
    batch_import(topic_generator)


def load_qrels():
    print(">>> Importing qrels")
    batch_import(qrel_generator)


def main():
    parser = ArgumentParser(description="Load the dataset to a database.")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub command help"
    )

    seed = subparsers.add_parser("seed", help="seed the database")
    seed.set_defaults(func=seed_database)

    import_documents = subparsers.add_parser(
        "documents", help="Add documents to database"
    )
    import_documents.set_defaults(func=load_documents)

    import_topics = subparsers.add_parser("topics", help="Add topics to database")
    import_topics.set_defaults(func=load_topics)

    import_qrels = subparsers.add_parser("qrels", help="Add qrels to database")
    import_qrels.set_defaults(func=load_qrels)

    args = parser.parse_args()
    args.func()  # Call the function associated with the selected subprogram


if __name__ == "__main__":
    main()

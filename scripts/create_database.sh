#!/bin/bash
python -m src.load_database seed
python -m src.load_database documents
python -m src.load_database topics
python -m src.load_database qrels
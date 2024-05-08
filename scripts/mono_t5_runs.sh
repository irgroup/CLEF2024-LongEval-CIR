#!/bin/bash

python -m systems.BM25+monoT5 --index t5

python -m systems.BM25+monoT5 --index t4
python -m systems.BM25+monoT5 --index t4 --topics t5

python -m systems.BM25+monoT5 --index t3
python -m systems.BM25+monoT5 --index t3 --topics t4
python -m systems.BM25+monoT5 --index t3 --topics t5


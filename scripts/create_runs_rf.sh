#!/bin/bash

python -m systems.BM25+relevance_feedback --index t3 --topics t3 --history t2 t1 t0
python -m systems.BM25+relevance_feedback --index t4 --topics t4 --history t3 t2 t1 t0
python -m systems.BM25+relevance_feedback --index t5 --topics t5 --history t3 t2 t1 t0

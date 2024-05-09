#!/bin/bash

# python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t0_T-t3
# python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t0_T-t4
# python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t0_T-t5
# echo "Done with D-t0"

python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t1_T-t3
python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t1_T-t4
python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t1_T-t5
echo "Done with D-t1"

python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t2_T-t3
python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t2_T-t4
python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t2_T-t5
echo "Done with D-t2"

# python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t3_T-t3
# python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t3_T-t4
# python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t3_T-t5
# echo "Done with D-t3"

# python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t4_T-t4
# python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t4_T-t5
# echo "Done with D-t4"

# python src/extend_runs.py documents --run data/results/trec/CIR_BM25_D-t5_T-t5
# echo "Done with D-t5"
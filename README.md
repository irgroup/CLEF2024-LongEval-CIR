# CLEF2024-LongEval-CIR

| Data split | identifier |
| --- | --- |
| 2023 training set | t_0 / WT |
| 2023 test set | t_1 / ST (2022_07) |
| 2023 test set | t_2 / LT (2022_09) |
| 2024 training set | t_3 / 2023_01 |
| 2024 test set | t_4 / 2023_06 |
| 2024 test set | t_5 / 2023_08 |

**Indices:**
- [ ] 1x small index based on relevant documents t_0-t_3 and Jüri's SQLite database
- [ ] 1x large index based on t_3
- [ ] 1x large index based on t_4
- [ ] 1x large index based on t_5

**Submission:** Rankings based on t_4 and t_5 

_Baselines:_
BM25(t_3)
BM25(t_4)
BM25(t_5)

_Naive filters:_
- BM25 of t_4 + Remove "non-relevant" documents from t_0 to t_3
- BM25 of t_5 + Remove "non-relevant" documents from t_0 to t_3

_(Pseudo) relevance feedback:_

Three runs based on PRF

(Hybrid RF approach) six (intermediate) topics files to produce three runs based on RF and PRF
Two topic sets for t_3: one with topics overlaps in t_0-t_2 and t_3, the other one for new topics in t_3
Two topic sets for t_4: one with topics overlaps in t_0-t_3 and t_4, the other one for new topics in t_4
Two topic sets for t_5: one with topics overlaps in t_0-t_3 and t_5, the other one for new topics in t_5

Afterwards, merge run files 

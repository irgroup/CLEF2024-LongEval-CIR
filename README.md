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
- [x] 1x large index based on t_3
- [x] 1x large index based on t_4
- [x] 1x large index based on t_5

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

- [x] (Hybrid RF approach) six (intermediate) topics files to produce three runs based on RF and PRF
- Two topic sets for t_3: one with topics overlaps in t_0-t_2 and t_3, the other one for new topics in t_3
- Two topic sets for t_4: one with topics overlaps in t_0-t_3 and t_4, the other one for new topics in t_4
- Two topic sets for t_5: one with topics overlaps in t_0-t_3 and t_5, the other one for new topics in t_5

Afterwards, merge run files 



## Experiments on t3:
### BM25 baselie
| P_10 | bpref | ndcg |
|---|---|---|
| 0.1624 | 0.4373 | 0.3638 |

### Naive filters:
Filter d, q pairs that are marked not relevant in previous sub-collection(s)

|filter from | P_10 | bpref | ndcg |
|---|---|---|---|
| **t3** | _0.1798_ | _0.7784_ | _0.3851_ |
| t2         | 0.1595 | 0.4415 | 0.3586 |
| t2, t1     | 0.1577 | 0.4411 | 0.3553 |
| t2, t1, t0 | 0.157  | 0.439  | 0.3528 |

> no improvement, effectiveness decreases with more filters



### BM25 + time fuse
- lost one topic because no document overlap between t3 and t2 in this ranking

| $\lambda$ | $\overline{\tau}$ | P_10 | bpref | ndcg |
|---|---|---|---|---|
| 0.501 | 0.0789 | 0.1652 | 0.437 | 0.3666 |
| 0.5004641588833613 | 0.1459 | 0.1635 | 0.4371 | 0.3653 |
| 0.5002154434690032 | 0.2631 | 0.163 | 0.437 | 0.3649 |
| 0.5001 | 0.4268 | 0.1627 | 0.4372 | 0.3641 |
| 0.5000464158883361 | 0.592 | 0.1627 | 0.4373 | 0.3639 |
| 0.5000215443469003 | 0.7306 | 0.1624 | 0.4374 | 0.3639 |
| 0.50001 | 0.8274 | 0.1624 | 0.4373 | 0.3638 |
| 0.5000046415888336 | 0.888 | 0.1624 | 0.4373 | 0.3638 |
| 0.50000215443469 | 0.9201 | 0.1624 | 0.4373 | 0.3638 |
| 0.500001 | 0.9378 | 0.1624 | 0.4373 | 0.3638 |

### BM25 + Filter Fuse
- Lost more topics

| $\overline{\tau}$ | P_10 | bpref | ndcg |
|---|---|---|---|
|0.00424 | 0.4218 | 0.1062 | 0.2914 | 


> P_10 is super high

### BM25 + qrel boost
Boos all relevant docs based on one or more qrels by the same lambda
|$\lambda$ | history | P_10 | bpref | ndcg |
|---|---|---|---|---|
| 1.001 | t2 | 0.1625 | 0.4375 | 0.3643 |
| 1.01  | t2 | 0.1651 | 0.4398 | 0.367 |
| 1.1   | t2 | 0.1742 | 0.4479 | 0.379 |
| 1.2   | t2 | 0.1766 | 0.4485 | 0.3805 |
| 1.3   | t2 | 0.1776 | 0.4485 | 0.3813 |
| 1.4   | t2 | 0.1778 | 0.4487 | 0.3815 |
| 1.5   | t2 | 0.1781 | 0.4491 | 0.3818 |
| 1.6   | t2 | 0.1784 | 0.4492 | 0.382 |
| 1.9   | t2 | 0.1788 | 0.4493 | 0.3822 |
| 2.5   | t2 | 0.1788 | 0.4493 | 0.3822 |
| 3     | t2 | 0.1788 | 0.4493 | 0.3822 |

> results improve over BM25. 
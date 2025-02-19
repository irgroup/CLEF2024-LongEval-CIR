# CLEF2024-LongEval-CIR


## Abstract
> This work reports our participation in the retrieval task of the second LongEval lab iteration at CLEF 2024. As
part of this year’s contribution, we analyze to which extent prior relevance signals on the document level and
term level can be used to improve the retrieval effectiveness. In order to exploit these kinds of signals, we fetch
corresponding document identifiers pointing to the same document in the different dataset slices of all timestamps.
Based on several heuristics, we submit and evaluate a total of five systems that either follow our previous year’s
methodology or that combine baseline rankings with prior relevance signals. Our evaluations provide insights to
which extent these signals can be used but let us also conclude with several recommendations for future work.
Most notably, we envision a companion resource that ties together all slices of the dataset by unified document
identifiers to have a better understanding of more rigorous data splits and to avoid potential data leakage that
might affect the evaluation of (deep) learning-based systems.


## Citation
```
@inproceedings{clef/prior_signals,
  author       = {J{\"{u}}ri Keller and
                  Timo Breuer and
                  Philipp Schaer},
  title        = {Leveraging Prior Relevance Signals in Web Search},
  booktitle    = {Working Notes of the Conference and Labs of the Evaluation Forum {(CLEF}
                  2024), Grenoble, France, 9-12 September, 2024},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {3740},
  pages        = {2396--2406},
  publisher    = {CEUR-WS.org},
  year         = {2024},
}
```

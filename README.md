## Overview
This code is modified from:
https://www.kaggle.com/dskswu/topic-modeling-bert-lda
### Prepare Workspace
* Create virtual environment (called ptorch):
    * run `conda env create --file ldabert.yml`
    * `conda activate ptorch`
* Install additional dependencies:
    * start python and run:
        ```
        import nltk
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
        nltk.download('punkt')
        exit()
      ```
* Run Jupyter From Within venv:
    * `jupyter notebook`
    * Open up `/examples/bert_lda_tm.ipynb`
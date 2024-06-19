---
title: Get tag 2
emoji: 🏷️
app_file: api.py
license: MIT
URL: get-tag.streamlit.app
---

[![Build, install, lint, test and format](https://github.com/sycod/get_tag2/actions/workflows/main.yaml/badge.svg)](https://github.com/sycod/get_tag2/actions/workflows/main.yaml)

# [➡️ API endpoint here ⬅️](https://get-tag.streamlit.app)

# What for?

This API is used to **predict tags** from a user input in a **StackOverFlow question context**.

The input is a text question, composed of a **title** and a **body**.  
It can include code, however it will be excluded by preprocessing.

Click the **prediction button** to **predict tags** related with this question.

# Contents

``` 
├── LICENSE
├── Makefile
├── README.md
├── api.py
├── config
│   ├── exclude_set.pkl
│   └── keep_set.pkl
├── favicon.ico
├── models
│   ├── w2v_cbow_lrovr_classifier.pkl
│   └── w2v_cbow_vectorizer
├── requirements.txt
├── src
│   ├── __pycache__
│   │   └── utils.cpython-311.pyc
│   └── utils.py
└── tests
    ├── __pycache__
    │   ├── test_eda.cpython-311-pytest-8.2.2.pyc
    │   ├── test_models.cpython-311-pytest-8.2.2.pyc
    │   ├── test_scrap_and_clean.cpython-311-pytest-8.2.2.pyc
    │   └── test_utils.cpython-311-pytest-8.2.2.pyc
    └── test_utils.py
```

# Installation

> Even though installation steps are the same, **following commands are for Unix OS**: for Windows users, see [how to install and run WSL](https://learn.microsoft.com/fr-fr/windows/wsl/install) in order to use a stable development OS 😉

1. **ensure to have Make** installed  
➡️ run `sudo apt install -y build-essential` (also installs other essential tools used along with Make)
2. **clone this repository** (HTTPS or SSH, see [GitHub documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository))
3. in this new folder, **create a new virtual environment**  
➡️ run `python -m venv .venv`
4. **activate** this environment  
➡️ run `source .venv/bin/activate`  
5. once activated, **install required packages** (listed in the requirements.txt file)  
➡️ run `make install`
6. **ready** to go!  
**prove it** by running unit tests:  
➡️ run `make test`
7. **run API** on a local server, easy peezy:
➡️ run `make run_ask_api`  
(which uses `streamlit run ask_api.py` in console)

# Technical notes

- This model uses **Word2Vec CBOW** embedder as vectorizer and **Logistic Regression OneVsRest** as classifier
- It was trained on **10k high quality StackOverFlow questions** over the past 4 years

# Tips

- Preprocessing discards many **frequent and usual words** plus **HTML tags** and **code snippets** from user sentences and may result to a too small final input.  
An error message can thus be displayed.
- Also note that the **model is trained for english language** input and may result in weird predictions in other cases.
- If model **can't find** any of the input words in trained data, it will display a **'no suggestion' message**
- If you see the **main page reloading at each run**, your browser doesn't allow the 'session state' management.  
Try using a **less secure browser for a full experience**, such as Chrome (which doesn't split storage and network states between websites)
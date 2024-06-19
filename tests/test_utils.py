"""Test module for utils.py"""

import pytest
import dill as pickle
from gensim.models import Word2Vec
import numpy as np

from src.utils import check_length
from src.utils import preprocess_doc
from src.utils import w2v_vect_data
from src.utils import lr_prediction
from src.utils import predict_tags


# NLTK downloads (just once, not downloaded if up-to-date)
import nltk
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)


# load necessary data
VECTORIZER_URI = "models/w2v_cbow_vectorizer"
CLASSIFIER_URI = "models/w2v_cbow_lrovr_classifier.pkl"
KEEP_SET_URI = "config/keep_set.pkl"
EXCLUDE_SET_URI = "config/exclude_set.pkl"
with open(KEEP_SET_URI, "rb") as f:
    keep_set = pickle.load(f)
with open(EXCLUDE_SET_URI, "rb") as f:
    exclude_set = pickle.load(f)
with open(CLASSIFIER_URI, "rb") as f:
    classifier = pickle.load(f)
vectorizer = Word2Vec.load(VECTORIZER_URI)

TOO_SHORT_BEFORE_PP = "less than 5 words"
TOO_SHORT_AFTER_PP = "more than 5 <pandas numpy> but not enough"
RAW_TITLE = "merge dataframes in python"
RAW_BODY = "how can i merge 2 dataframes? <image> i tried using <b>python pandas</b> library or <b>numpy arrays</b>"
RAW_INPUT = RAW_TITLE + " " + RAW_BODY
PP_INPUT = "merge dataframes python merge dataframes python pandas numpy arrays"
PRED_TAGS_SET = {"validation", "git", "pandas", "python", "pre-commit-hook"}


@pytest.mark.parametrize(
    "input_doc", [TOO_SHORT_BEFORE_PP, TOO_SHORT_AFTER_PP, RAW_INPUT]
)
@pytest.mark.parametrize("length", [5])
def test_check_length(input_doc, length):
    """Test src.utils.check_length function"""
    if len(input_doc.split(" ")) >= length:
        assert check_length(input_doc) == True
    else:
        assert check_length(input_doc) == False


@pytest.mark.parametrize("document", [RAW_INPUT])
@pytest.mark.parametrize("keep_set", [keep_set])
@pytest.mark.parametrize("exclude_set", [exclude_set])
def test_preprocess_doc(document, keep_set, exclude_set):
    """Test src.utils.preprocess_doc function"""
    assert preprocess_doc(document, keep_set, exclude_set) == PP_INPUT


@pytest.mark.parametrize("input_clean", [PP_INPUT])
@pytest.mark.parametrize("vectorizer", [vectorizer])
@pytest.mark.parametrize("classifier", [classifier])
def test_predict_tags(input_clean, vectorizer, classifier):
    """Test src.utils.predict_tags function"""
    predictions = set(predict_tags(input_clean, vectorizer, classifier).split(" "))
    assert predictions == PRED_TAGS_SET

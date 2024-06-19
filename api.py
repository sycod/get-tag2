"""Predict tags from a StackOverflow-like question through this API."""

import os
import dill as pickle
import logging
from flask import Flask, request, jsonify
from gensim.models import Word2Vec
import nltk

# home made
from src.utils import check_length
from src.utils import preprocess_doc
from src.utils import predict_tags

# CONFIG
# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# NLTK downloads (just once, not downloaded if up-to-date)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
# ML tools
KEEP_SET_URI = "config/keep_set.pkl"
EXCLUDE_SET_URI = "config/exclude_set.pkl"
VECTORIZER_URI = "models/w2v_cbow_vectorizer"
CLASSIFIER_URI = "models/w2v_cbow_lrovr_classifier.pkl"

# load keep set (for preprocessing)
logging.info("⚙️  Loading keep set...")
if os.path.exists(KEEP_SET_URI):
    with open(KEEP_SET_URI, "rb") as f:
        keep_set = pickle.load(f)
    logging.info("✅ Keep set loaded")
else:
    logging.warning("⚠️ No keep set found ⚠️")

# load exclude set (for preprocessing)
logging.info("⚙️  Loading exclude set...")
if os.path.exists(EXCLUDE_SET_URI):
    with open(EXCLUDE_SET_URI, "rb") as f:
        exclude_set = pickle.load(f)
    logging.info("✅ Exclude set loaded")
else:
    logging.warning("⚠️ No exclude set found ⚠️")

# load vectorizer
logging.info("⚙️  Loading vectorizer...")
if os.path.exists(VECTORIZER_URI):
    vectorizer = Word2Vec.load(VECTORIZER_URI)
    logging.info("✅ Vectorizer loaded")
else:
    logging.warning("⚠️ No vectorizer found ⚠️")

# load classifier
logging.info("⚙️  Loading classifier...")
if os.path.exists(CLASSIFIER_URI):
    with open(CLASSIFIER_URI, "rb") as f:
        classifier = pickle.load(f)
    logging.info("✅ Classifier loaded")
else:
    logging.warning("⚠️ No classifier found ⚠️")


app = Flask(__name__)


@app.route("/")
def home_page():
    return "<h1>Get tags (from where you once asked for)</h1>"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.get_json(force=True)["user_input"]
        logging.debug(f"{user_input = }")

        # check user input
        if not check_length(user_input):
            message = "⚠️  Input length is too short"
            predicted_tags = None
            logging.warning(message)
        else:
            # preprocess input
            input_clean = preprocess_doc(user_input, keep_set, exclude_set)
            logging.info(f"\nClean input: {input_clean}")

            # check preprocessed input length before predict
            if not check_length(input_clean):
                message = "⚠️  Length is too short after preprocessing: check input"
                predicted_tags = None
                logging.warning(message)
            else:
                message = None
                predicted_tags = predict_tags(input_clean, vectorizer, classifier)
            # log infos
            logging.info(f"\nPredicted tags: {predicted_tags}")

        return jsonify({"message": message, "predicted_tags": predicted_tags})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)

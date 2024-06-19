"""Streamlit simple interface."""

import streamlit as st
import requests


# CONFIG
# API_URL = "https://credit-f7b3b911ecd1.herokuapp.com/predict"
API_URL = "https://localhost:13370/predict"
# placeholders
TITLE_PLACEHOLDER = "example: pandas merge with Python >3.5"
BODY_PLACEHOLDER = """example:
How do I add NaNs for missing rows after a merge?
How do I get rid of NaNs after merging?
I've seen these recurring questions asking about various facets of the pandas merge functionality, the aim here is to collate some of the more important points for posterity."""
TAGS_PLACEHOLDER = "see five predicted tags here"
# check session state
if "title_input" not in st.session_state:
    st.session_state.title_input = ""
if "body_input" not in st.session_state:
    st.session_state.body_input = ""
if "predicted_tags" not in st.session_state:
    st.session_state.predicted_tags = TAGS_PLACEHOLDER
if "message" not in st.session_state:
    st.session_state.message = None


# update session state on inputs
def update_title():
    st.session_state.title_input = st.session_state.title


def update_body():
    st.session_state.body_input = st.session_state.body


# main function, triggered with button
def click_button():
    """Actions to perform when button clicked"""
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    user_input = st.session_state.title_input + "\n" + st.session_state.body_input
    post_data = {"user_input": user_input}

    print("\n\n********** DEBUG ***********")
    print(f"{user_input = }")
    print(f"{post_data = }")
    print("********** DEBUG ***********\n\n")

    response = requests.post(API_URL, headers=headers, json=post_data, timeout=5)

    try:
        result = response.json()
        st.session_state.predicted_tags = result["predicted_tags"][0]
        st.session_state.message = result["message"][0]
    except Exception as e:
        st.error(f"Error getting prediction: {e}")

    return st.session_state.predicted_tags


# GUI
st.set_page_config(
    page_title="Get tags (from where you once asked for)",
    page_icon="favicon.ico",
    layout="centered",
)
st.write("# Tags prediction")
st.write(
    "Predict 5 tags from a StackOverflow-like question title and / or body fields:"
)

# user input
st.text_input(
    "Title", placeholder=TITLE_PLACEHOLDER, key="title", on_change=update_title
)
st.text_area(
    "Body", placeholder=BODY_PLACEHOLDER, height=160, key="body", on_change=update_body
)

# predictions
st.button(
    "⬇️  Predict tags  ⬇️",
    type="primary",
    use_container_width=True,
    on_click=click_button,
)
# display message if no prediction (e.g. input is too short)
if st.session_state.predicted_tags is not None:
    st.write("#### :blue[{}]".format(st.session_state.predicted_tags))
else:
    st.write("#### :red[{}]".format(st.session_state.message))

# info and tips
st.divider()
st.write("## ℹ️ TIPS")
st.markdown(
    """- Preprocessing discards many **frequent and usual words** plus **HTML tags** and **code snippets** from user sentences and may result to a too small final input.  
    An error message can thus be displayed."""
)
st.write(
    "- Also note that the **model is trained for english language** input and may result in weird predictions in other cases."
)
st.write(
    "- If model can't find any of the input words in trained data, it will display a 'no suggestion' message"
)
st.markdown(
    """- If you see the main page reloading at each run, your browser doesn't allow the 'session state' management.  
    Try using a less secure browser, such as Chrome which doesn't split storage and network states between websites"""
)

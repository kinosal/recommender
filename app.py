"""Streamlit app to recommend anything based on personal photos."""

import logging

import streamlit as st

import rekognition as rek
import gpt
import bedrock as bed

# Configure logger
logging.basicConfig(format="\n%(asctime)s\n%(message)s", level=logging.INFO, force=True)


# Define functions
def detect_objects(image_files, model):
    """Detect objects in images."""
    labels = set()
    for image_file in image_files:
        hashed_image, hashed_image_name = rek.hash_and_scale_image(
            mode="file", image_file=image_file
        )
        if not rek.find_image(hashed_image_name):
            image_url = rek.upload_image(
                mode="file", image_name=hashed_image_name, image_file=hashed_image
            )
        else:
            image_url = f"https://{rek.BUCKET}.s3.amazonaws.com/{hashed_image_name}"
        if model == "Amazon Rekognition (Faster)":
            objects = rek.detect_labels(hashed_image_name)
        elif model == "GPT-4 Vision (Slower)":
            objects = gpt.detect_labels(image_url)
        labels.update(objects)
    logging.info("\n" + ", ".join(labels))
    return labels


def generate_recommendations(topic, image_files, vision_model, text_model):
    """Generate recommendations."""
    st.session_state.error = ""

    if not topic:
        st.session_state.error = "Please enter a topic"
        return

    if not image_files:
        st.session_state.error = "Please upload at least one image"
        return

    if len(image_files) > 10:
        st.session_state.error = "Please upload a maximum of 10 images"
        return

    if (
        image_files
        and topic
        and (
            st.session_state.image_files != image_files
            or st.session_state.vision_model != vision_model
        )
    ):
        st.session_state.image_files = image_files
        st.session_state.vision_model = vision_model
        with spinner_placeholder:
            with st.spinner("Analyzing your photos..."):
                st.session_state.labels = detect_objects(image_files, vision_model)

    if st.session_state.labels and topic:
        with spinner_placeholder:
            with st.spinner("Generating your personal recommendations..."):
                if text_model == "GPT-3.5":
                    st.session_state.recommendations = gpt.recommend(
                        st.session_state.labels, topic, "gpt-3.5-turbo"
                    )
                elif text_model == "GPT-4":
                    st.session_state.recommendations = gpt.recommend(
                        st.session_state.labels, topic, "gpt-4-1106-preview"
                    )
                elif text_model == "Llama-2":
                    st.session_state.recommendations = bed.recommend(
                        st.session_state.labels, topic, "meta.llama2-13b-chat-v1"
                    )
                logging.info(f"\nTopic: {topic}")
                logging.info("\n" + st.session_state.recommendations)


# Configure Streamlit page and state
st.set_page_config(page_title="Recommender", page_icon="🤖")
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "image_files" not in st.session_state:
    st.session_state.image_files = []
if "vision_model" not in st.session_state:
    st.session_state.vision_model = ""
if "text_model" not in st.session_state:
    st.session_state.text_model = ""
if "labels" not in st.session_state:
    st.session_state.labels = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = ""
if "error" not in st.session_state:
    st.session_state.error = ""

# Render Streamlit page
st.title("Recommend me anything")
st.markdown(
    "This mini-app recommends any activity by analyzing a few personal photos. It uses Amazon [Rekognition](https://aws.amazon.com/rekognition/image-features/) or OpenAI's [GPT Vision](https://platform.openai.com/docs/models) to detect objects in the uploaded images, and OpenAI's [GPT Text](https://platform.openai.com/docs/models) or Meta's Llama 2 on Amazon [Bedrock](https://aws.amazon.com/bedrock/llama-2/) to generate respective recommendations.\n\nYou can find the code on [GitHub](https://github.com/kinosal/recommender) and the author on [Twitter](https://twitter.com/kinosal)."
)

topic = st.text_input(
    label="Topic", placeholder="Books, movies, travel destinations, ..."
)

vision_model = st.selectbox(
    label="Vision Model",
    options=["Amazon Rekognition (Faster)", "GPT-4 Vision (Slower)"],
)

text_model = st.selectbox(
    label="Text Model", options=["GPT-3.5", "GPT-4", "Llama-2"]
)

image_files = st.file_uploader(
    label="Upload up to 10 personal photos",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

st.button(
    label="Generate recommendations",
    type="primary",
    on_click=generate_recommendations,
    args=(topic, image_files, vision_model, text_model),
)

spinner_placeholder = st.empty()

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.recommendations:
    st.markdown("""---""")
    st.text_area(
        label="Your recommendations", value=st.session_state.recommendations, height=300
    )

if st.session_state.recommendations:
    st.text_area(
        label="Your image labels", value=", ".join(st.session_state.labels), height=200
    )

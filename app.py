"""Streamlit app to recommend anything based on personal photos."""

import streamlit as st

import rekognition as rek
import gpt


# Define functions
def detect_objects(image_files):
    """Detect objects in images."""
    labels = set()
    for image_file in image_files[:5]:
        image_name = rek.hash_and_scale_image(mode="file", image_file=image_file)
        if not rek.find_image(image_name):
            image_url = rek.upload_image(image_name)
        else:
            image_url = f"https://{rek.BUCKET}.s3.amazonaws.com/{image_name}"
        if model == "Amazon Rekognition":
            objects = rek.detect_labels(image_name)
        elif model == "GPT-4 Vision":
            objects = gpt.detect_labels(image_url)
        labels.update(objects)
    return labels


def generate_recommendations(topic, image_files):
    """Generate recommendations."""
    st.session_state.error = ""

    if not topic:
        st.session_state.error = "Please enter a topic"
        return

    if not image_files:
        st.session_state.error = "Please upload at least one image"
        return

    if len(image_files) > 5:
        st.session_state.error = "Please upload a maximum of five images"
        return

    if image_files and topic and st.session_state.image_files != image_files:
        st.session_state.image_files = image_files
        with spinner_placeholder:
            with st.spinner("Analyzing your photos..."):
                st.session_state.labels = detect_objects(image_files)

    if st.session_state.labels and topic:
        with spinner_placeholder:
            with st.spinner("Generating your personal recommendations..."):
                st.session_state.recommendations = gpt.recommend(
                    st.session_state.labels, topic
                )


# Configure Streamlit page and state
st.set_page_config(page_title="Recommender", page_icon="🤖")
if "image_files" not in st.session_state:
    st.session_state.image_files = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = ""
if "error" not in st.session_state:
    st.session_state.error = ""

# Render Streamlit page
st.title("Recommmend me anything")
st.markdown(
    "This mini-app recommends any activity by analyzing a few personal photos. It uses Amazon [Rekognition](https://aws.amazon.com/rekognition/image-features/) or OpenAI's [GPT-4 Vision](https://platform.openai.com/docs/models) to detect people, objects, places, and sentiment in the uploaded images, and OpenAI's [GPT-4 Text](https://platform.openai.com/docs/models) to generate respective recommendations. You can find the code on [GitHub](https://github.com/kinosal/recommender) and the author on [Twitter](https://twitter.com/kinosal)."
)

topic = st.text_input(
    label="Topic", placeholder="Books, movies, travel destinations, ..."
)

model = st.selectbox(
    label="Vision Model", options=["Amazon Rekognition", "GPT-4 Vision"]
)

image_files = st.file_uploader(
    label="Upload up to 5 personal photos",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

st.button(
    label="Generate recommendations",
    type="primary",
    on_click=generate_recommendations,
    args=(topic, image_files),
)

spinner_placeholder = st.empty()

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.recommendations:
    st.markdown("""---""")
    st.text_area(
        label="Your recommendations", value=st.session_state.recommendations, height=300
    )

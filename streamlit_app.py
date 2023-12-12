import streamlit as st
from fastai.vision.all import *
from PIL import Image

title = "Group 5 Big Data Project"

# Load the fastai learner model
model_path = "./models/fastai_model.pkl"
learn = load_learner(model_path)

st.set_page_config(
    page_title=title,
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(title)

if st.sidebar.button("EDA"):
    pass

uploaded_file = st.sidebar.file_uploader("Choose an image and test model", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", width=300)

    # Button to make predictions
    if st.button("Make Prediction"):
        # Resize the image to the size expected by the model
        img_resized = image.resize((224, 224))

        # Convert the PIL Image to a fastai Image
        img_fastai = PILImage.create(img_resized)

        # Perform inference using the loaded model
        pred_class, pred_idx, probabilities = learn.predict(img_fastai)

        # Display predictions
        st.write(f"Predicted Class: {pred_class}")
        st.write(f"Prediction Probability: {probabilities[pred_idx].item():.4f}")

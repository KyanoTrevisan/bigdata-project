import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pandas as pd
import os
import random
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
st.write("For our Big Data project, we chose to make a classifier which predicts between big cats (Jaguars, Lions, Snow Leopards, Leopards and Tigers).")
st.write("For a quick EDA you can click the EDA button on the sidebar. This will produce 1 random image from each class and also it will show the data distribution between classes.")
st.write("If you wish to test out the model, you can upload a file via the file uploader on the siderbar. This will result in your selected picture to show on the screen with the 'predict' button.")
if st.sidebar.button("EDA"):
    st.subheader("Exploratory Data Analysis (EDA)")

    # Specify the path to your dataset
    dataset_path = "./datasets/big-cats/"

    # Initialize variables to store EDA results
    num_images_per_class = []
    class_labels = []

    # Create columns for displaying images side by side
    columns = st.columns(3)  # Adjust the number of columns as needed

    # Loop through each subdirectory (animal category) in the dataset
    for i, animal_category in enumerate(os.listdir(dataset_path)):
        animal_category_path = os.path.join(dataset_path, animal_category)
        if os.path.isdir(animal_category_path):
            class_labels.append(animal_category)

            # Count the number of images in each class
            num_images = len(os.listdir(animal_category_path))
            num_images_per_class.append(num_images)

            # Get a list of image files in the category
            image_files = os.listdir(animal_category_path)
            random.shuffle(image_files)  # Shuffle the list

            # Display one random image of each class with label in a column
            with columns[i % len(columns)]:
                random_image_path = os.path.join(animal_category_path, image_files[0])
                random_image = Image.open(random_image_path)
                st.image(random_image, caption=f"{animal_category}", width=150)

    # Create a DataFrame to display the EDA results
    eda_df = pd.DataFrame({"Animal Category": class_labels, "Number of Images": num_images_per_class})
    st.dataframe(eda_df)


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

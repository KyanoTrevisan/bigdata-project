# Big Cat Species Classifier

## Overview
This project is an implementation of an image classifier designed to identify various big cat species. The classifier is built using Fastai on top of PyTorch, leveraging transfer learning with convolutional neural networks. It was developed as part of the BigData course assignment.

## Team Members
- [Emre Ekici](https://github.com/emre-ekici2)
- [Salih Ekici](https://github.com/SalihEkici)
- [Kyano Trevisan](https://github.com/KyanoTrevisan)

## Dataset
The dataset comprises approximately 200 images each of the following big cat species:
- Jaguar (Panthera onca)
- Leopard (Panthera pardus)
- Lion (Panthera leo)
- Snow Leopard (Panthera uncia)
- Tiger (Panthera tigris)

These images were collected using a custom script `scraper.py` which automates the process of scraping images from Google.

## Models
We experimented with various architectures including AlexNet, VGG, and ResNet to determine the most effective model for our classification task. The models were trained and validated on our dataset, and their performance was compared based on accuracy and error rates.

## Files
- `scraper.py`: Script used for scraping images from Google to build the dataset.
- `big_cat_classifier.ipynb`: Jupyter notebook containing the full pipeline for training and evaluating the image classifier. [Google Colab Page Placeholder]

## Usage
To collect images for the dataset, install the requirements.txt (`pip install -r requirements.txt`) and run the `scraper.py` script.

For training and evaluating the model:
1. Open the `big_cat_classifier.ipynb` in [Google Colab](https://colab.research.google.com/drive/1Qc24NvHu5K3NUWwyhR6MEfNtjyV-eGql?usp=sharing).
2. Runtime > Run All
3. Evaluate models

## Results
The models' performance is documented within the Jupyter notebook, including confusion matrices and per-class accuracy metrics. After rigorous testing, our model outperformed the baseline established by Google's Teachable Machine in identifying the specified big cat species.

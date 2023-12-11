import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests

my_keywords = ["tiger panthera tigris", "lion panthera leo", "jaguar panthera onca", "leopard panthera pardus", "snow leopard panthera uncia"]

# Define the split ratio
split_ratio = 0.8  # 80% for training, 20% for testing

# Define the number of images to download for each keyword
images_per_keyword = 300


def get_img_urls(keyword):
    driver = webdriver.Chrome()
    driver.get(f"https://www.google.com/search?q={keyword}&tbm=isch")
    time.sleep(5)
    for i in range(images_per_keyword // 25):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        time.sleep(0.5)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    driver.quit()
    img_tags = soup.findAll('img')
    img_urls = []
    # get the tags of the html elements
    for img_tag in img_tags:
        if img_tag.has_attr('data-src'):
            img_url = img_tag['data-src']
            img_class = img_tag.get('class')

            if 'no69gc' not in img_class:
                img_urls.append(img_url)
    return img_urls


def download_image(url, keyword, index, target_dir):
    response = requests.get(url)
    filename = f"{keyword}_{index + 1}.jpg"  # Create the filename
    file_path = os.path.join(target_dir, filename)
    with open(file_path, "wb") as file:
        file.write(response.content)


# Set the 'test' directory
test_dir = "datasets/animals/test"
# Create 'training' directory if it doesn't exist
training_dir = "datasets/animals/training"
if not os.path.exists(training_dir):
    os.makedirs(training_dir)

for keyword in my_keywords:
    img_urls = get_img_urls(keyword)
    num_images = len(img_urls)

    # Limit the number of images to download for this keyword
    num_to_download = min(images_per_keyword, num_images)
    num_training = int(num_to_download * split_ratio)

    for i, img_url in enumerate(img_urls[:num_to_download]):
        if i < num_training:
            target_dir = os.path.join(training_dir, keyword)
        else:
            target_dir = os.path.join(test_dir, keyword)

        # Create the subdirectory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        download_image(img_url, keyword, i, target_dir)

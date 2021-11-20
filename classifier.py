# Importing all modules
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

# Getting the data
# Into X, all the images are being stored. In the Y, all the letters from classes are being stored.
X = np.load('image.npz')['arr_0']
y = pd.read_csv('data.csv')["labels"]

# Print
print(pd.Series(y).value_counts())

# All the labels/classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] 
nclasses = len(classes)

# Splitting the data for training and testing

# Split into 25% and 75% from 10,000
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

# Scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

# Defining the clf with logistic regression
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

# Predicting image function (accuracy)
def get_prediction(image):
    # Opening the images
    im_pil = Image.open(image)
    # Converting to black and white colors
    image_bw = im_pil.convert('L')
    # Resizing the images 
    image_bw_resized = image_bw.resize((22,30), Image.ANTIALIAS)
    pixel_filter = 20
    # Picking the minimum pixels (percentile), converting them (clip -> number to each image), then picking the maximum pixels (max)
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)

    # Convert the image to an array
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,660)

    # Predict the accuracy
    test_pred = clf.predict(test_sample)
    # Return the value
    return test_pred[0]
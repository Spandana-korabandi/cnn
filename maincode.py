#Import TensorFlow and Keras for deep learning
import tensorflow as tf  # TensorFlow is an open-source machine learning library
from tensorflow import keras  # Keras is an open-source software library that provides a Python interface for artificial neural networks

# Import specific Keras modules for building a neural network
from tensorflow.keras.models import Sequential  # Sequential is the model type for linear stacking of layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D  # Import various layers types for neural network construction

# Import tools for preprocessing image data
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # ImageDataGenerator is used to augment and preprocess image data

# Import sklearn metrics for model evaluation
from sklearn.metrics import classification_report, confusion_matrix  # Tools for evaluating the performance of the model

# Import PIL and Path for image and path handling
from PIL import Image  # PIL is the Python Imaging Library, used for opening, manipulating, and saving many different image file formats
from pathlib import Path  # Pathlib is used for easy handling of file paths

# Import scipy, os, numpy, and matplotlib for additional functionalities
import scipy  # SciPy is a Python-based ecosystem of open-source software for mathematics, science, and engineering
import os  # OS module in Python provides a way of using operating system dependent functionality
import numpy as np  # NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices
import matplotlib.pyplot as plt  # Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy

# Import PyTorch ImageFolder and transforms for image loading and transformation
from torchvision.datasets import ImageFolder  # ImageFolder is a generic data loader where the images are arranged in this way by PyTorch
import torchvision.transforms as T  # Transforms are common image transformations available in PyTorch

print("Done with library declaration, Current version of Tensorflow is: ", tf.__version__)

# Set the directory path for the dataset
data_dir = Path('/content/drive/MyDrive/DataSets_ECE510')  # Define the path to the directory containing the images

# Define transformations for the dataset
transformer = T.Compose([T.Resize((32, 32)), T.ToTensor()])  # Compose transformations: resize images to 32x32 and convert them to tensors

# Create a dataset from the images in the directory applying the transformations
dataset = ImageFolder(data_dir, transform = transformer)  # Load images from the specified directory with the defined transformations

# Display the names of the classes in the dataset
print(dataset.classes)  # Print the list of class names extracted from the subdirectories

# Prepare to display the distribution of classes in the dataset
fig = plt.figure()  # Create a new figure for plotting
ax = fig.add_axes([0,0,1,1])  # Add axes to the figure

# Hardcoded counts of images in each class (ideally, this should be dynamically calculated)
counts = [74,90,82,79]  # Specify the number of images in each class

# Plot a bar chart showing the class distribution
ax.bar(dataset.classes, counts)  # Create a bar chart with class names and their corresponding image counts

# Add a title to the plot and display it
plt.title('Class Distribution')  # Set the title of the plot
plt.show()  # Display the plot

PATH_TRAIN = r"/content/drive/MyDrive/DataSets_ECE510"
PATH_TEST = r"https://drive.google.com/drive/folders/1ApKAM6XRHd_E3mMaORkIa-Bjdg17UxlZ?usp=sharing"
class_names = ['Aluminum_Cans', 'Plastic', ' cigrattee_Butts ', 'Food_waste']

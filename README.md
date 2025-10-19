Machine Learning Implementations: Regression and KNN

This repository contains two Jupyter notebooks demonstrating the implementation of fundamental machine learning algorithms from scratch and comparing them with their Scikit-learn counterparts.

1. Regression Analysis on California Housing Data (Regression.ipynb)

This notebook focuses on predicting median house values from the California Housing dataset using various linear regression techniques.

Overview

The primary goal is to implement and evaluate different methods for solving linear regression problems. The notebook covers:

Data preprocessing, including data shuffling and feature scaling (standardization).

Splitting the data into training, validation, and test sets.

Implementation of multiple regression techniques.

Models and Methods

Direct Solution (Normal Equation): A closed-form solution for finding the optimal weights without iterative optimization.

Batch Gradient Descent: A manual implementation of the gradient descent algorithm to iteratively find the optimal model weights.

Regularized Gradient Descent (Ridge and Lasso):

Ridge (L2 Regularization): Implemented manually to penalize large coefficients and prevent overfitting.

Lasso (L1 Regularization): Implemented using Scikit-learn to perform feature selection by shrinking some coefficients to zero.

Scikit-learn's SGDRegressor: Used as a baseline to compare the performance of the manual implementations.

Results

The notebook concludes with a comprehensive comparison of all models on the test set, evaluating their performance based on Mean Squared Error (MSE) and Mean Absolute Error (MAE).

2. K-Nearest Neighbors (KNN) Classification (KNN.ipynb)

This notebook provides a detailed walkthrough of the K-Nearest Neighbors (KNN) algorithm, applying it to the MAGIC Gamma Telescope dataset for particle classification (gamma vs. hadron).

Overview

This project implements the KNN algorithm from scratch and validates its performance against the Scikit-learn library. Key steps include:

Data loading and preprocessing, including data shuffling, standardization, and down-sampling to handle class imbalance.

Manual implementation of the KNN classifier using NumPy.

Hyperparameter tuning to find the optimal value of 'k' (the number of neighbors) by evaluating accuracy on a validation set.

Comparison with Scikit-learn's KNeighborsClassifier.

Implementations

Manual KNN Classifier: A from-scratch implementation that calculates Euclidean distances, finds the 'k' nearest neighbors, and predicts the class label based on a majority vote.

Scikit-learn KNeighborsClassifier: The standard library implementation used to verify the results of the manual classifier.

Evaluation

Both implementations are evaluated on the test set, with performance measured by:

Accuracy

Precision, Recall, and F1-score (per class)

Confusion Matrix

How to Run the Notebooks

Dependencies

To run these notebooks, you will need Python 3 and the following libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

You can install them using pip:

pip install pandas numpy matplotlib seaborn scikit-learn



Usage

Clone this repository.

Make sure the datasets (California_Houses.csv and telescope_data.csv) are in a datasets/ subdirectory.

Open and run the Jupyter notebooks (Regression.ipynb and KNN.ipynb).

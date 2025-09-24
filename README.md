KNN Classifier for Heart Disease Prediction
----------------------------------------------------

This project implements a K-Nearest Neighbors (KNN) classifier from scratch (no scikit-learn or scipy) to predict the likelihood of heart disease based on the UCI Heart Disease dataset.

The pipeline includes:

Data preprocessing (handling missing values, normalization, one-hot encoding)

Train/validation/test split (60/20/20)

Euclidean distance calculation

KNN neighbor search and majority voting classification

Evaluation metrics (confusion matrix, accuracy, precision, recall, F1 score)

Data Cleaning & Preprocessing
------------------------------------------

Normalizes continuous features with z-score

Fills missing values with label-based mean/mode

Encodes categorical features using one-hot encoding

Custom KNN Implementation

Euclidean distance calculated manually

Supports multiple values of k

Majority voting via numpy.bincount

Evaluation Tools

Confusion matrix

Accuracy, precision, recall, F1-score

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Naive Bayes Classifier for Heart Disease Prediction
--------------------------------------------------------

This project implements a Naive Bayes Classifier from scratch (no scikit-learn or scipy) to predict the likelihood of heart disease using the UCI Heart Disease dataset.

The pipeline includes:

Data preprocessing (handling missing values, normalization, one-hot encoding)

Train/validation/test split (60/20/20)

Probability estimation for continuous and categorical features

Features
--------------

Data Cleaning & Preprocessing

Normalizes continuous features with z-score

Handles missing values by filling with mean/mode based on label

Encodes categorical features with one-hot encoding

Custom Naive Bayes Implementation

Assumes conditional independence between features

Uses Gaussian likelihoods for continuous features

Uses frequency-based probabilities for categorical features

Evaluation Tools

Confusion matrix

Accuracy, precision, recall, F1-score

Classification using Bayes’ Theorem

Evaluation metrics (confusion matrix, accuracy, precision, recall, F1 score)

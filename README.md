# Mammogram Mass Prediction

This project aims to predict whether a mammary mass is benign or malignant using machine learning.

## Description

We use the public dataset [Mammographic Masses](https://archive.ics.uci.edu/dataset/161/mammographic+mass) available in the UCI repository. This dataset contains 961 instances of masses detected in mammograms and includes the following attributes:

- **BI-RADS**: Assessment of diagnostic reliability (ordinal scale from 1 to 5).
- **Age**: Age of the patient in years (integer).
- **Shape**: Shape of the mass (round=1, oval=2, lobular=3, irregular=4).
- **Margin**: Margin of the mass (circumscribed=1, microlobulated=2, obscured=3, ill-defined=4, spiculated=5).
- **Density**: Density of the mass (high=1, iso=2, low=3, fatty=4).
- **Severity**: Classification of diagnosis (benign=0, malignant=1).

The **BI-RADS** attribute will not be used for prediction, as it only assesses the confidence of the medical classification. The focus is on the attributes **Age**, **Shape**, **Margin**, and **Density** to build our model. The goal is to predict the **Severity** of the mass.

## Motivation

False positives in mammograms can lead to unnecessary medical procedures and cause anxiety for patients. The goal is to develop a machine learning model that can help reduce these false positives, providing a more accurate interpretation of mammogram results.

## Objective

Build and train a Multilayer Perceptron (MLP) neural network to classify mammary masses as benign or malignant based on the provided characteristics. The preprocessing of the data includes:

1. Cleaning missing data and handling outliers.
2. Normalizing the data.
3. Experimenting with different architectures, optimizers, and hyperparameter tuning.
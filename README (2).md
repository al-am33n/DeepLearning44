# Ex-4: Binary Classifier for MNIST

## Overview
In this exercise, you will implement a binary classifier to classify a single number from the MNIST dataset. You will be required to complete the functions provided in the Python files.

## Objectives
1. Define dense models with single and hidden layers.
2. Train a model to classify a single digit from the MNIST dataset.
3. Evaluate the performance of the trained model.

## The Assignment
You are required to complete the following functions in 'layered_model.py'. Your completed functions should pass the tests provided.
1. `define_dense_model_single_layer`: Define a dense model with a single layer.
2. `define_dense_model_with_hidden_layer`: Define a dense model with a hidden layer.
3. `fit_mnist_model_single_digit`: Train the model for a single digit classification.
4. `evaluate_mnist_model_single_digit`: Evaluate the performance of the trained model.

---

## Validating and Evaluating Your Results

### Online
1. After committing and pushing your code, check the mark on the top line (near the commit ID).
2. If some tests are failing, click on the ❌ to open up a popup, which will show details about the errors.
3. You can click the [Details]() link to see what went wrong. Pay special attention to lines with the words "Failed" or "error".

![screnshot](images/details_screenshot.png)

4. Near the bottom of the [Details]() page, you can see your score. Here are examples of 0/5 and 5/5:

![score](images/score.png) ![success](images/success.png)

5. When you achieve a perfect score, you will see a green checkmark near the commit ID.

![green](images/green.png)

### Locally
1. You can test your code locally by installing and running `pytest` (`pip install pytest` or `conda install pytest`).
2. Run the tests using the command `pytest` in your terminal. This will show the status of each test and any errors that occurred.

Good luck!
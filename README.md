# Galton Board Simulator and Naive Bayes Classifier

This repository contains two distinct components:

1. **Galton Board Simulator**: A simulation of a Galton board using a binomial distribution to model the falling of balls through pegs, demonstrating the central limit theorem.
2. **Naive Bayes Classifier**: A Naive Bayes classifier for classifying species of flowers in the Iris dataset, utilizing Gaussian distributions for feature modeling.
3. **Naive Bayes Trainer**: The training class for the Naive Bayes classifier that computes class priors and feature statistics for each class.
4. **DataWhitener**: A utility for normalizing or whitening the dataset, ensuring that the features have zero mean and unit variance.

---

## Galton Board Simulator

### Description

A Galton board is a mechanical device used to demonstrate the central limit theorem. Balls are dropped at the top and hit a series of pegs, bouncing randomly left or right as they fall. By the time they reach the bottom, the balls settle into various bins, and the distribution of balls in these bins follows a binomial or normal distribution.

This simulation runs multiple trials of balls being dropped, and the final position of each ball is recorded. The Probability Mass Function (PMF) is then calculated and visualized.

## Naive Bayes Classifier

### Description

The **Naive Bayes Classifier** is a probabilistic classifier based on Bayes' theorem. It assumes that the features are conditionally independent given the class label, which is why it is termed "naive." Despite the simplifications made by this assumption, the classifier often performs surprisingly well in many real-world applications, particularly when dealing with text classification and other high-dimensional datasets.

In this implementation, we use the **Gaussian Naive Bayes** approach, where each feature follows a Gaussian (normal) distribution. The classifier computes class priors and feature statistics (mean and variance) for each class, and during prediction, it calculates the likelihood of a sample belonging to each class using these statistics. The predicted class is the one with the highest posterior probability.

### Code Overview

1. **Training (NaiveBayesTrainer)**: The `NaiveBayesTrainer` class is responsible for training the model. It computes the class priors and the mean and variance of each feature for each class. These statistics are stored in a JSON file (`model.json`) to be used later during prediction.
   - The training process involves calculating the probability of each class based on its frequency in the training data (`class_priors`).
   - For each feature, we calculate the mean and variance for each class (`class_params`).
   - After training, the model is saved to a JSON file for future use.

2. **Prediction (NaiveBayesClassifier)**: The `NaiveBayesClassifier` class loads the pre-trained model (from the `model.json` file) and uses it to predict the class of new samples.
   - For each sample, the classifier calculates the log-probability of the sample belonging to each class. The feature values are modeled using a Gaussian distribution, and the log-probabilities are accumulated for each class.
   - The class with the highest log-probability is chosen as the predicted label.

3. **Testing**: The `test()` method in the `NaiveBayesClassifier` class allows us to evaluate the performance of the classifier on a test dataset. The test set contains labeled samples, and the classifier predicts the class for each sample. The accuracy of the classifier is then computed by comparing the predicted labels with the true labels.

### Example Output

After training the model, running the classifier on a test dataset will provide output similar to the following:



## Data Whitener

### Description

The **Data Whitener** is a process that transforms a dataset such that the transformed data has zero mean and a covariance matrix that is the identity matrix. This is typically achieved through a linear transformation of the original data.

In this implementation, we:
1. Compute the mean (E[X]) of the original dataset `X` and center the data by subtracting the mean.
2. Calculate the covariance matrix of the centered data.
3. Perform Singular Value Decomposition (SVD) on the covariance matrix to obtain the eigenvalues and eigenvectors.
4. Use the eigenvalues and eigenvectors to construct the whitening matrix `A`.
5. Compute the transformed data `W` using the whitening matrix `A` and bias vector `b`.
6. Finally, we check that the mean of `W` is zero and the covariance matrix of `W` is the identity matrix, as expected.

### Code Overview

1. **Mean and Covariance Calculation**: We compute the mean (`mu`) and covariance matrix (`COV_X`) of the original data `X`.
2. **Whitening Matrix Calculation**: Using the covariance matrix `COV_X`, we apply SVD to obtain the eigenvalues (`S`) and eigenvectors (`U`). The whitening matrix `A` is calculated using the formula:  
   `A = S^(-1/2) * U.T`
3. **Bias Vector Calculation**: We compute the bias vector `b` as `b = -A * mu`.
4. **Transformation**: The transformed data `W` is computed as `W = A * X + b`. This transformation ensures that the data has zero mean and the covariance matrix of `W` is the identity matrix.
5. **Results**: We print the mean and covariance of the transformed data to verify the whitening process.

### Example Output

After running the code, you should expect the following results:
- **Mean (E[X])**: A vector representing the mean of the original data.
- **Covariance Matrix (COV[X, X])**: The covariance matrix of the original data.
- **Whitening Matrix (A)**: The matrix used to transform the data.
- **Bias Vector (b)**: The vector that compensates for the shift in mean.
- **Transformed Data**: The mean of the transformed data should be close to zero, and the covariance matrix should approximate the identity matrix.




### Requirements

- Python 3.x
- NumPy
- Matplotlib


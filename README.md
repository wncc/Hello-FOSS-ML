# Hello-FOSS-ML

Hello There!
This project is a part of HELLO-FOSS: Celebration of Open Source by the Web and Coding Club. We will be focusing on building basic ML models for the MNIST data set. The Repo has been given the label of "Hacktoberfest". Refer to [Hello-FOSS](https://github.com/wncc/Hello-FOSS) for guidelines.

## Guidelines

Absolutely No Prerequisites for contributing to this Project.
We will be using Juniper Notebooks for our Project. If you are an absolute beginner in python have a look at [this](https://github.com/wncc/learners-space/tree/master/Python).

# 1) Handwritten Digit Recognizer

**MNIST** ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

### Problem Statement
You can do one or more of the following - 

### Task-1:

- A basic model of Neural Networks has been implemented in TensorFlow [Filename: MNIST_CNN].
- Your task is to improve the accuracy to 97% by changing the number of layers and/or adding convolutions.
- Do not worry if you are an absolute begineer as pseudo code has already been given in the comments.

### Task-2:

- Test data has been loaded and a function to plot this data has also been made [Filename: MNIST_PCA].
- Your task is to perform PCA on this data, for reference look at the examples [here](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
- Now that you have performed PCA, implement this data using a KNN or Logistic Regression or any other algorithm you prefer.

### Task-3:
- Code to get the training and testing data has already been written [Filename: MNIST_FIT].
- Your task is to train this data in the least time and with the best accuracy. Any algorithm can be used to do so, to implement algorithms in tensorflow refer to [this](https://www.tensorflow.org/tutorials) or [this](https://www.kaggle.com/learn/overview).


# 2) Titanic: Machine Learning from Disaster
This is the legendary Titanic ML problem – a good challenge for you to dive into Open Source through ML. The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered ''unsinkable'' RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: ''what sorts of people were more likely to survive?'' using passenger data (ie name, age, gender, socio-economic class, etc).

### Overview of Dataset
The data has been split into two groups:

training set (train.csv)
test set (test.csv)
The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the ''ground truth'') for each passenger. Your model will be based on ''features'' like passengers' gender and class. You can also use **feature engineering** to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include `gender_submission.csv`, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

### Problem Statement
You can choose to do one or more of the following tasks - 
1. **Exploratory Data Visualisation** - The notebooks already have the part included which provide analysis of main characteristics of data. You are welcome to add to this for visualising more trends.
2. **Build a Grader** - An `output.csv` file has been included in the folder with [titanic dataset](./titanic). It has the expected predictions corresponding to [test data](./titanic/test.csv) and your task is to write a Python fucntion which takes as input a csv file, compares to the expected output and prints the percentage of accuracy achieved.
3. **Hyperparameter tuning** - A neural network has been trained on the dataset in [this notebook](./Titanic%20neural%20network%20Tensorflow.ipynb) but it has very low accuracy. Your task is to tune the hyperparameters and improve the model. Some suggestions - modifying the optimizer, adding more hidden layers to the model or changing the dimension of layers, adding dropout, regularization etc.
4. **Back-Propogation in Numpy** - A neural network has been coded from scratch in Numpy on the dataset in [this notebook](./Titanic%20neural%20network%20Numpy%20from%20scratch.ipynb) but it is incomplete; it misses the function for calculating gradients during back-propogation. Your task is to complete the function for calculating gradients, train the model and output the predictions on test data to a csv file.
5. **Implementing algorithm in sklearn** - Apart from a neural network, there are many other ML algorithms that can be used to make predictions in this challenge. Notebooks for some of them have been put up but only with data reading and visualisation part completed. Your task is to write the code for implementing the specificied algortihm on the dataset using sklearn library and printing the predictions to a csv file. You can choose from any of the following or get started with one of your own! 
  * [Logistic Regression](./Titanic%20logistic%20regression%20Tensorflow.ipynb)
  * [Support Vector Machine](./Titanic%20SVM%20Tensorflow.ipynb)
  * [Random Forests](./Titanic%20Random%20Forest%20Tensorflow.ipynb)
  * [K-Nearest Neighbours](./Titanic%20KNN%20Tensorflow.ipynb)
  * [Gaussian Naive Bayes model](./Titanic%20GaussianNB%20Tensorflow.ipynb)
  
  Join our [Discord](https://discord.com/invite/mzhyrvS) for discussing your doubts.

***

<p align="center">Created with :heart: by <a href="https://www.wncc-iitb.org/">WnCC</a></p>



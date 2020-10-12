# Hello-FOSS-ML

Hello There!
This project is a part of HELLO-FOSS: Celebration of Open Source by the Web and Coding Club. We will be focusing on building basic ML models for the MNIST data set. The Repo has been given the label of "Hacktoberfest". Refer to [Hello-FOSS](https://github.com/wncc/Hello-FOSS) for guidelines.

## Guidelines

Absolutely No Prerequisites for contributing to this Project.
We will be using Jupyter Notebooks for our Project. If you are an absolute beginner in python have a look at [this](https://github.com/wncc/learners-space/tree/master/Python).
NOTE: before sending any pr change the name of the file to TASK(no. of the task)_(your initials).

# 1) Titanic: Machine Learning from Disaster
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
2. **Build a Grader** - An `output.csv` file has been included in the folder with [titanic dataset](./titanic). It has the expected predictions corresponding to [test data](./titanic/test.csv) and your task is to write a Python function which takes as input a csv file, compares to the expected output and prints the percentage of accuracy achieved.
3. **Hyperparameter tuning** - A neural network has been trained on the dataset in [this notebook](./Titanic%20neural%20network%20Tensorflow.ipynb) but it has very low accuracy. Your task is to tune the hyperparameters and improve the model. Some suggestions - modifying the optimizer, adding more hidden layers to the model or changing the dimension of layers, adding dropout, regularization etc.
4. **Back-Propagation in Numpy** - A neural network has been coded from scratch in Numpy on the dataset in [this notebook](./Titanic%20neural%20network%20Numpy%20from%20scratch.ipynb) but it is incomplete; it misses the function for calculating gradients during back-propagation. Your task is to complete the function for calculating gradients, train the model and output the predictions on test data to a csv file.
5. **Implementing algorithm in sklearn** - Apart from a neural network, there are many other ML algorithms that can be used to make predictions in this challenge. Notebooks for some of them have been put up but only with data reading and visualisation part completed. Your task is to write the code for implementing the specificied algortihm on the dataset using sklearn library and printing the predictions to a csv file. You can choose from any of the following or get started with one of your own! 
  * [Logistic Regression](./Titanic%20logistic%20regression%20Tensorflow.ipynb)
  * [Support Vector Machine](./Titanic%20SVM%20Tensorflow.ipynb)
  * [Random Forests](./Titanic%20Random%20Forest%20Tensorflow.ipynb)
  * [K-Nearest Neighbours](./Titanic%20KNN%20Tensorflow.ipynb)
  * [Gaussian Naive Bayes model](./Titanic%20GaussianNB%20Tensorflow.ipynb)
  
# 2) MNIST and Fashion MNIST

**MNIST** ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

The datasets can be downloaded [here](https://github.com/zalandoresearch/fashion-mnist/tree/master/data).

### Problem Statement
You can do one or more of the following - 

**Task-1:**
- A basic model of Neural Networks has been implemented in TensorFlow [Filename: MNIST_CNN].
- Your task is to improve the accuracy to 97% by changing the number of layers and/or adding convolutions.
- Do not worry if you are an absolute begineer as pseudo code has already been given in the comments.

**Task-2:**

- Test data has been loaded and a function to plot this data has also been made [Filename: MNIST_PCA].
- Your task is to perform PCA on this data, for reference look at the examples [here](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
- Now that you have performed PCA, implement this data using a KNN or Logistic Regression or any other algorithm you prefer.

**Task-3:**
- Code to get the training and testing data has already been written [Filename: MNIST_FIT].
- Your task is to train this data in the least time and with the best accuracy. Any algorithm can be used to do so, to implement algorithms in tensorflow refer to [this](https://www.tensorflow.org/tutorials) or [this](https://www.kaggle.com/learn/overview).

**Task-4:**
- Generative Adversarial Networks, or GANs for short, are an approach to generative modeling using deep learning methods, such as convolutional neural networks. An incomplete DCGAN has been implemented on the MNIST dataset in Tensorflow. 
- Your task is to define and the Discriminator and Generator Model and the create GIFs for visualising generation of Handwritten Digits from Random Noise. 
- A very detailed explanation of GANs has also been given in the [notebook](./DCGan_MNIST.ipynb)

**Task-5:**
- Build a CNN model for Fashion MNIST dataset. An incomplete model has been build for you in this [notebook](./CNN%20Fashion%20mnist.ipynb)
- Your task is to complete the CNN model, train it on the dataset and visualise the accuracy and loss.
- Subsequently make predictions on the test set and plot the confusion matrix.

**Task-6:**
- Train a GAN on the Fashion MNIST dataset. Code for Conditional GAN on it has been included in the [notebook](./Conditional%20GAN%20Fashion%20MNIST%20.ipynb)
- Your task is to complete the generator and discriminator functions, train the model and save the generated images.
- Subsequenlty, you have to create a markdown file with display of those generated images and a short summary explaining the theory behind Conditional GANs (Be as creative as you can :)

**Task-7:**
- All the notebooks we have provided are in Tensorflow as of now. This is a flexible task wherein you have to implement the above in PyTorch. 
- You can choose to implement any model of your choice on MNIST or Fashion MNIST.


  
  Join our [Discord server](https://discord.com/invite/mzhyrvS) for discussing your doubts.

***

<p align="center">Created with :heart: by <a href="https://www.wncc-iitb.org/">WnCC</a></p>



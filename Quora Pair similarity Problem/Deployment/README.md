# Quora-Question-Pair-Similarity-Deployed-on-Flask

<h1> <center>Quora Question-Piar Similarity</center></h1>

**NOTE:** [Link](https://question-pair-similarity.herokuapp.com/) of the deployed model.

# 1. Problem Statement
- Identify which questions asked on Quora are duplicates of questions that have already been asked.
- This could be useful to instantly provide answers to questions that have already been answered.
- We are tasked with predicting whether a pair of questions are duplicates or not.

**Note:** [Data Source](https://www.kaggle.com/c/quora-question-pairs/data)

# 2. Machine Learning Probelm
1. It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.
* **Performance Metric**
  * __<a href = 'https://www.kaggle.com/c/quora-question-pairs/overview/evaluation'>log-loss__</a>
  * Binary Confusion Matrix
 * We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.

# 3. EDA
**Note:** Always refers to the notebook, to know in deatils about each plots/outcomes.
## 3.1 Class Distribution
<img src="./Images/class dirstribution.png" alt="Class Distribution"/>
This plot show the class distribution of dataset. From above plot it can be concluded that this dataset is imbalanced because the number of datapoints for class-1 (Similar) is less than class-0 (Not similar)

## 3.2 Unique v/s Repeated Question
<img src="./Images/unique_vs_repeated_questions.png"  
alt="Unique vs Repeated-Questions"/>

Above plot shows that the number of unique questions is almost five time the number of repeated question.

## 3.3 Desinged Features
There are 24 desinged features (hand crafted features). Analysis of all features has been done in the Quora_EDA.ipynb notebook.

### 3.3.1 **word_share** Feature
<img src="./Images/voilin and box plot.png" alt="Voilin & Box plot of word_share feature">

Above features looks important (but not too much) because distribution plot is not completely overlapping

# 4. ML Models
Various models has been tried like **Logistic Regression, Linear SVM and GBDT** with **BoW, TF-IDF and W2V** vectorizer along with hand-crafted features. But out of all these **GBDT with TFIDF vectorizer** gave the best performance.
For all other models training, calibration and tunning look at the notebook,

## 4.1 GBDT
The dataset was large, so I used *lightgbm* library for training the GBDT model. Gridseach method is used to find the best combination of **max_depth and n_estimators** hyperparamters of the model. Rest other hyperparameters are kept as deafult.
 <img src="./Images/tunning plot.png"  
alt="Tunning Plot for GBDT"  
style="float: left; margin-right: 10px;" />
In above plot,  each cell of hearmap shows the **log-loss** value **(max_depth, estimator)** pair. Best hyperparameter pair is: max_depth=10, n_estimators=500.

## 4.2 Performance Summary of All Models
<img src="./Images/perforamce summary.png"  
alt="Performance Summary"  
style="float: left; margin-right: 10px;" />

# Requirements
* python == 3.6
* flask == 1.1.1
* gunicorn == 20.0.4
* scikit-learn == 0.22.1
* numpy == 1.16.4
* pandas == 0.24.2
* lightgbm == 2.2.3
* pickle == 4.0
* scipy == 1.2.0
* nltk == 3.4.3
* fuzzywuzzy == 0.17.0
* distance == 0.1.3

# File Details
<img src="./Images/tree structure of files.png"  alt="Tree Structure of Files">

## Flask API (folder)
It contains all required file to build a **UI**. This folder is used to deploy the model on **Heroku**.
[Link](https://question-pair-similarity.herokuapp.com/) of the deployed model.

## *.ipynb* Extension
With this extension, files are IPython Notebooks. File name *Quora_EDA.ipynb* has exploratory data anaysis. In exploartory data analysis various distribution plots, barplots, t-SNE plots etc are plotted. In this file some additional features are designed. In *quora_vectorizing_and_models.ipynb* file first TF-IDF and TFIDF-W2V vectorization are done then final TF-IDF and TFIDF-W2V features are created by merging vectorized features with designed features in *Quora_EDA.ipynb* file. Logistic regression, linear SVM and GBDT (with LightGBM) machine learning models are applied. For each model hyper-parameter tunning has been done. At the end all model's log-loss values are compared.

## 'test.py' and 'feature.py'
These files is for testing the model.

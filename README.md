# Tiktok User Ban Analysis

## Overview 
This project aims to provide an estimate the author ban status of a TikTok user based on several features such as the amount of likes, comments and claim status etc. 

A classification model was created via data found on Kaggle. Data was subsequently explored in terms their relationship to the dependent variable in question. Classifications models were compared, tuned, and pickled for deployment via Flask and Docker.

## Data Description 
The [Kaggle dataset](https://www.kaggle.com/datasets/yakhyojon/tiktok/download?datasetVersionNumber=1) can be found here. The zip file contains around ~14k user video information. 
- Independent Variables (variables that we are going to explore to predict price): video information, verfication status etc.
- Dependent Variable (the variable that we are interested in predicting): author ban status

> `EDA.ipynb` contains most the information about the dataset and exploration

## Modeling
Modeling was done using a simple Logistic Regression model, Random Forest, and AdaBoost and comparing two important metrics: AUC, ROC and accuracy scores. Random Forest was ultimately chosen as the final model due to having a higher performance metrics. The model was subsequently tuned to increase performance. The final model was pickled for use in deployment.

> `train.py` trains the and pickles the final model

## Local Deployment

`pipenv` and `docker` were used for deployment for local machines

1) Clone the repo to your local computer via `git clone <INSERT LINK HERE>`
2) `cd` into the repo
3) Build the docker image: `docker build -t <name_your_app> .`
4) Next run: `docker run -p 9696:9696 <name_your_app>`
5) A testing script was provided with `Test-Values.ipynb` or use `http://0.0.0.0:9696/predict`
6) Edit the value in the test script and see the prediction!

To elaborate, since we have used `pipenv` to create a virtual env and having `docker` build an image based on those requirements we can containerize our application and ensure that the Flask app is served and the port exposed correctly 

# Data-Science
This repository contains two independent but complementary Data Science projects developed using Python in Google Colab. Each notebook follows a different methodology and objective — one focused on the CRISP-DM process for regression forecasting, and the other implementing K-Means clustering from scratch and with Scikit-learn.

*📂 Project 1: CRISP_DM.ipynb — Data Science Lifecycle Implementation*
Overview
This notebook applies the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology to build a regression-based forecasting model using real-world accounting transaction data spanning three years.

Objectives
- Understand and prepare a financial dataset with multiple account types.
- Explore data through visualization and statistical analysis.
- Build and evaluate several regression models to forecast future transaction amounts.
- Deploy the best-performing model using IBM Watson Machine Learning.

Dataset
A CSV file containing 4,212 financial records from 2019–2021 with the following fields:
Year, Month, Cost Centre, Account, Account Description, Account Type, and Amount.

Methodology (CRISP-DM Phases)
Business Understanding — Define the forecasting goal for financial transactions.
Data Understanding — Load and analyze the structure, types, and distribution of data.
Data Preparation — Transform categorical variables with one-hot encoding and clean fields.

Modeling — Train and compare multiple regressors:
Random Forest Regressor
Gradient Boosting Regressor
Ridge, Lasso, and ElasticNet
Best model: Gradient Boosting with R² ≈ 0.52 and MAE ≈ 216.7.

Evaluation — Compare model accuracy using R² and MAE metrics.

Deployment — Store and deploy the model on IBM Cloud via Watson Machine Learning API.

Key Libraries
pandas, numpy, matplotlib, seaborn, scikit-learn, ibm-watson-machine-learning

Results Summary
- The dataset was successfully prepared for machine learning with categorical and numerical encoding.
- Gradient Boosting achieved the best predictive performance.
- The model was deployed to IBM Watson for real-time predictions.

*📂 Project 2: KMeansClustering.ipynb — Unsupervised Learning*
Overview
This project focuses on implementing the K-Means clustering algorithm manually and using Scikit-learn, with applications on synthetic data, the Digits dataset, and the Iris dataset.

Objectives
- Understand the mechanics of K-Means clustering.
- Implement the algorithm step by step without libraries.
- Apply clustering to real-world datasets and visualize results.
- Evaluate model performance using confusion matrices.

Sections
1. Manual Implementation — Build K-Means from scratch using NumPy and SciPy.
2. Generate a small 2D dataset and iteratively update centroids.
3. Predict cluster membership for new data points.
4. K-Means with Scikit-learn — Apply clustering to the Digits dataset.
5. Visualize 10 clusters representing digits (0–9).
6. Predict test data groupings and visualize sample images.
7. Iris Dataset Clustering — Apply K-Means to classic Iris data.
8. Identify 3 clusters corresponding to species (setosa, versicolor, virginica).
9. Compute confusion matrix to assess cluster–class alignment.

Key Libraries
numpy, pandas, scipy, matplotlib, seaborn, scikit-learn

Results Summary
- Successfully implemented K-Means logic manually, showing iterative centroid adjustment.
- Applied clustering to high-dimensional data (Digits, Iris).
- Achieved meaningful class–cluster correspondence (e.g., Iris confusion matrix [[12 0 0], [0 12 1], [0 3 10]]).

*⚙️ How to Run*

Clone this repository:
git clone https://github.com/yourusername/Data-Science-Projects.git
cd Data-Science-Projects


Open each notebook in Google Colab or Jupyter:
CRISP_DM.ipynb for regression and deployment workflow.
KMeansClustering.ipynb for clustering analysis.

Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn ibm-watson-machine-learning

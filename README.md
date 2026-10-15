# Workout Recovery Score Prediction Using Supervised Machine Learning

## Project Overview

This project aims to predict workout recovery scores using supervised machine learning. It uses data from WHOOP wearable tracking devices to analyze the relationship between recovery scores and different physiological, sleep, and workout-related features.

The dataset is divided into training, validation, and testing sets. Each set undergoes data preprocessing using a defined pipeline. Four tree-based regression models are evaluated using six regression metrics to identify the best-performing default model.

The selected model is then tuned using three different hyperparameter optimization methods. The results are compared to evaluate whether hyperparameter tuning improves model performance. The findings are presented through data visualizations and used to develop conclusions and recommendations.

## Dataset

The dataset contains 100,000 records from 286 Whoop fitness band users. It includes recovery scores and other features related to sleep, heart rate, heart rate variability, workout activity, and user characteristics.

The target variable is `recovery_score`, which ranges from 0 to 100.

Dataset source: [Whoop_Fitness_Dataset [100K]](https://www.kaggle.com/datasets/likithagedipudi/whoop-fitness-dataset)

## Regression Models

Four tree-based regression models are evaluated using their default configurations:

- **Decision Tree Regressor** - A tree-based model that makes predictions by recursively splitting the data based on feature thresholds.
- **HistGradientBoosting Regressor** - A gradient boosting model that builds multiple trees sequentially while using histogram-based binning to improve training speed.
- **XGBoost Regressor** - A gradient boosting model that builds trees sequentially and uses regularization and other optimizations to improve prediction performance and training efficiency.
- **LightGBM Regressor** - A gradient boosting model designed for fast training and low memory usage by using histogram-based learning and leaf-wise tree growth.

## Hyperparameter Tuning Methods

The best-performing default model undergoes hyperparameter tuning using the following methods:

- **Bayesian Optimization** – Uses the results of previous trials to identify promising hyperparameter combinations and guide the next trials toward better model performance. 
- **Successive Halving** - Tests multiple hyperparameter combinations using limited resources, removes the weaker candidates, and gives more resources to the better-performing ones.
- **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** – An evolutionary optimization method that searches for better hyperparameter combinations by adjusting its search based on previous results and the distribution of promising candidate. 

## Evaluation Metrics

Model performance is evaluated using six regression metrics:

- **Root Mean Squared Error (RMSE)** - Measures the average prediction error while giving more weight to larger errors. Lower values indicate better performance
- **Mean Squared Error (MSE)** -  Measures the average squared difference between the predicted and actual values. Lower values indicate better performance.
- **Mean Absolute Error (MAE)** - Measures the average absolute difference between the predicted and actual values. Lower values indicate better performance.
- **Mean Absolute Percentage Error (MAPE)** - Measures the average prediction error as a percentage of the actual values. Lower values indicate better performance.
- **Maximum Error (ME)** - Measures the largest absolute difference between a predicted value and its actual value. Lower values indicate better performance.
- **R² Score** - Measures how well the model explains the variation in the target variable. Higher values indicate better performance.

## Result Visualization

Result visualizations are used to compare model performance and examine prediction errors. These include:

- **Bar graphs** - for comparing evaluation metrics across models
- **Line graph** - for presenting performance comparisons
- **Residiual plots** - for examining the differences between actual and predicted values

 

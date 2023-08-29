# Sales Prediction using Multiple Linear Regression

This project demonstrates the implementation of a Multiple Linear Regression model to predict sales based on advertising expenditures in TV, Radio, and Newspaper.

## Project Overview

In marketing, understanding how advertising investments in various channels impact sales is crucial. This project aims to build a predictive model that helps businesses make informed decisions regarding their advertising budgets.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- pandas
- numpy
- scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn
```

## Usage

1. **Data Preparation**: Ensure you have a dataset named `Salesdata.csv` containing the following columns: 'TV', 'Radio', 'Newspaper', and 'Sales'. Make sure the data is clean and free from missing values.

2. **Running the Model**: Run the `multiple_linear_regression.py` script to train the model and make predictions. The script includes the following steps:
   
   - Data loading and preprocessing.
   - Splitting the data into training and testing sets.
   - Training a Multiple Linear Regression model.
   - Making predictions on the test set.
   - Evaluating the model using RMSE (Root Mean Squared Error) and R-squared (R^2) metrics.
   - Making predictions for new advertising expenditures.

3. **Model Serialization**: The trained model is serialized using `pickle` and saved as `model.pkl`. This allows for easy deployment and reusing the model without retraining.

4. **Results**: The model's performance metrics (RMSE and R^2) are displayed in the console, and predictions for new data are also provided.

## Customization

- You can customize the script by changing the random state for the train-test split in the script if you need different splits for reproducibility.
- Consider feature selection techniques or experimenting with different sets of features for modeling.



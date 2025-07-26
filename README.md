# 📈 Google Stock Price Prediction

A machine learning project focused on predicting Google stock prices using various regression techniques. The analysis compares the performance of different models on real historical stock data.

---


## 🎯 Objective
To predict Google's stock closing prices using regression models. This helps understand the relationship between stock features and closing price, and evaluate how well different algorithms generalize.

---

## 📚 Dataset
- **Source**: [Yahoo Finance](https://www.kaggle.com/datasets/alirezajavid1999/google-stock-2010-2023/data)
- **Target Variable**: Closing Price (`Close`)
- **Features Used**:
  - `Open`, `High`, `Low`, `Volume`, etc.
  - Z-score normalization, mean-centering and scaling by factor of standard deviation has been used for data normalization.

---

## 🛠️ Tools & Libraries
- Python, Jupyter Notebook
- `pandas`, `numpy`, `matplotlib`
- `scikit-learn`: for preprocessing, modeling, and evaluation

---

## ⚙️ Models Implemented
- Linear Regression
- Ridge Regression (`alpha=0.1`, `0.5`, `1`)
- Lasso Regression (`alpha = 0.1`, `0.001`, `0.000001`)
- ElasticNet Regression(`alpha = 0.1`, `0.001`, `0.00001`)
- BUt when the one of the iterative optimization algorithms , gradient descent has been applied to optimize the loss function in Linear regression it outperforms(Stochastic Gradient Descent (SGD))

---

## 🧹 Data Preprocessing

- **Missing Value Check**:  
  Checked for null (`NaN`) values in the dataset using `df.isnull().sum()`. This ensures the dataset is clean and ready for analysis. Missing values were handled appropriately if found.

- **Data Overview**:  
  - Displayed basic information including dataset **shape**, **data types**, and **statistical summary** using `df.shape`, `df.dtypes`, and `df.describe()`.
  - Previewed the first few records using `df.head()` to get a quick look at the data.

- **Multicollinearity Check**:  
  - Created a **correlation matrix** and visualized it using `seaborn.heatmap()` to identify multicollinearity between features.
  - This step helped in detecting highly correlated features that could impact model stability.

- **Feature Scaling**:  
  - Applied **StandardScaler** from `sklearn.preprocessing` to normalize the features. This brings all features to a common scale, which is crucial for models like Ridge, Lasso, and SGD.

- **Train-Test Splitting**:  
  - Used `train_test_split()` to divide the dataset into **training** and **testing** sets.
  - This separation ensures that model performance can be evaluated on unseen data.



---

## 📊 Evaluation Metrics
Each model is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score (Coefficient of Determination)

---

## 🏆 Best Model
> The Ridge Regression model with appropriate regularization showed a balance between low error and high R² on test data.

---

## 📈 Visualizations
- Loss function over epoch has been plotted to see how the loss is decreasing or increasing over epochs.

---

## 🔍 Key Takeaways
- Regularization (Ridge/Lasso) helps prevent overfitting.
- Feature scaling is critical for gradient-based models.
- With MBGD optimization algorithm it outperforms.


---

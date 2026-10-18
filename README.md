# SC5002-Lab-1

## 1. Project Overview

This project explores supervised machine learning using the California Housing dataset. The objective is to predict median house values using numerical features related to housing characteristics, population, income, and geographical location.

The experiment compares Linear Regression with Ridge Regression using different regularisation strengths (alpha values). Five-fold cross-validation is used to evaluate model performance, and a bar chart visualises the results.

## 2. Dataset

The California Housing dataset is obtained using `fetch_california_housing()` from scikit-learn.

- **Dataset:** California Housing
- **Sample size:** 1,000 randomly selected records
- **Target variable:** `MedHouseVal`
- **Sampling random state:** 42

### Input Features

The model uses the following eight numerical features:

| Feature | Description |
|---|---|
| `MedInc` | Median income |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Latitude |
| `Longitude` | Longitude |

The target variable, `MedHouseVal`, represents the median house value.

## 3. Methodology

### 3.1 Data Preparation

The dataset is loaded and 1,000 records are randomly sampled using `random_state=42`. The target variable is separated from the input features before model training.

### 3.2 Feature Scaling

`StandardScaler` is used to standardise the numerical features so that they are on comparable scales. This is particularly relevant for Ridge Regression because regularisation penalises the size of model coefficients.

### 3.3 Train-Test Split

The data is split into:
- 80% training data
- 20% test data

The split uses `random_state=42` to make the split reproducible.

### 3.4 Model Training and Evaluation

Four model configurations are compared:
1. Linear Regression
2. Ridge Regression with alpha = 0.1
3. Ridge Regression with alpha = 10.0
4. Ridge Regression with alpha = 1000.0

Five-fold cross-validation is performed on the training data. The mean R² score is used to evaluate how well each model explains the variation in median house values. A higher R² score indicates better performance under this evaluation.

### 3.5 Visualisation

A bar chart compares the mean five-fold cross-validation R² scores of the four models, making the effect of different Ridge regularisation strengths easier to observe.

## 4. Results and Discussion

The recorded experimental results are shown below.

| Model | Alpha | Mean 5-Fold CV R² |
|---|---:|---:|
| Linear Regression | N/A | 0.6457 |
| Ridge Regression | 0.1 | 0.6457 |
| Ridge Regression | 10.0 | 0.6443 |
| Ridge Regression | 1000.0 | 0.3574 |

### Key Findings

- **Linear Regression and Ridge (alpha = 0.1):** Both achieved a mean R² score of 0.6457, indicating similar performance in this experiment.
- **Ridge (alpha = 10.0):** Performance decreased slightly to 0.6443.
- **Ridge (alpha = 1000.0):** Performance dropped substantially to 0.3574.

The results illustrate that increasing the alpha value does not necessarily improve model performance. Ridge Regression applies a penalty to large coefficients. When alpha becomes excessively large, the coefficients may be shrunk too strongly, preventing the model from capturing important relationships in the data. This can lead to underfitting.

In this experiment, a very high alpha value performed considerably worse than the baseline Linear Regression model.

## 5. How to Run the Notebook

### Requirements

- Python 3
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Google Colab or Jupyter Notebook

### Running in Google Colab

1. Open the `.ipynb` notebook in Google Colab.
2. Run the code cells sequentially, or select **Runtime → Run all**.
3. Allow the California Housing dataset to be downloaded if necessary.
4. Review the printed cross-validation scores.
5. View the generated model comparison chart.

The notebook uses fixed random states for sampling and the train-test split to support reproducibility.

## 7. Repository Contents

- `README.md` — Project overview, dataset description, methodology, results, and instructions.
- `SC5002_Lab_Assignment_Part2.ipynb` — Python notebook containing data preparation, model training, cross-validation, and visualisation.

## 8. Limitations

This experiment uses a random sample of 1,000 records rather than the full California Housing dataset. Results may differ when using a different sample or evaluation setup.

Although a train-test split is created, the current notebook reports cross-validation scores on the training data rather than a final evaluation score on the held-out test set.

In addition, the scaler is fitted before the train-test split and cross-validation. A more rigorous implementation could use a scikit-learn `Pipeline` to fit preprocessing separately within each training fold, reducing the risk of data leakage.

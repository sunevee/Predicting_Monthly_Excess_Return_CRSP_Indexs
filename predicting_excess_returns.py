# %% [markdown]
# <a href="https://colab.research.google.com/github/demilade27/Predicting-Monthly-Excess-Returns-of-Market-Index/blob/main/Predicting_Monthly_Excess_Returns_of_Market_Index.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# <a href="https://colab.research.google.com/github/demilade27/Predicting-Monthly-Excess-Returns-of-Market-Index/blob/main/Predicting_Monthly_Excess_Returns_of_Market_Index.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Pre-requisite
# 
# 
# 

# %% [markdown]
# ## Import Libraries
# 

# %%
# Comment out the pip requirement 
# !pip install -r requirements.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Final
import sklearn
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_ccf
from statsmodels.tsa.stattools import acf, pacf
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# %% [markdown]
# ## Load The Data

# %%
df = pd.read_csv('https://raw.githubusercontent.com/demilade27/Predicting-Monthly-Excess-Returns-of-Market-Index/d2f2cb8478612fa4e8fd4e87628375d44f6cb72e/data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
DF_FINAL: Final = df 

# %% [markdown]
# # Analyse the dataset

# %% [markdown]
# ## Generate descriptive statistics

# %%
df.describe()

# %% [markdown]
# ## Check for null or zero values
# *Analysis*
# ---
# There are no null values in the dataset
# Analysing the dataset values there are zero values
# * INFL: There are 239 zero values showing signs of Deflationary Stagnation
# * DE:
# * LTR:
# * TMS:
# * DFR:

# %%
print(df.isnull().sum())
print(df.duplicated().sum())
print((df == 0).sum())

# %% [markdown]
# ## Analyse Data Skewness

# %%
df.skew()

# %% [markdown]
# 
# ## Data Visualization
# 

# %% [markdown]
# ### Correlation analysis

# %% [markdown]
# #### Correlation analysis of dataset

# %%
correlation_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,fmt=".2f",cmap="coolwarm")
plt.title('Correlation Heatmap')

# %% [markdown]
# #### Correlation analysis with R

# %%
target_correlation = correlation_matrix[['R']]
plt.figure(figsize=(10,8))
sns.heatmap(target_correlation,annot=True,fmt=".2f",cmap="coolwarm")
plt.title('Correlation Heatmap with R')

# %% [markdown]
# ### Residual plot
# 

# %%
sns.pairplot(df,y_vars=['R'],x_vars=df.select_dtypes(include='number').columns,kind='reg')
plt.suptitle("Pairwise Scatterplots with Fitted Lines")
plt.show()

# %% [markdown]
# ### Box Plot
# This is the analysis of the skewness of the data

# %%
plt.figure(figsize=(12, 6))
df.boxplot()
plt.title('Box Plot of Training Data Features')
plt.xticks(rotation=45, ha='right')
plt.show()

# %% [markdown]
# ### Autocorrelation and Partial Autocorrelation Analysis

# %% [markdown]
# #### 12 Month lag

# %%
plot_acf(df['mr'],lags=12)
plot_pacf(df['mr'],lags=12)
plt.show()

# %% [markdown]
# #### 24 Month lag

# %%
plot_acf(df['mr'],lags=24)
plot_pacf(df['mr'],lags=24)
plt.show()

# %% [markdown]
# # Pre-processing

# %% [markdown]
# ## Data Spliting

# %%
split_date='2019-01-01'
train_data=df[df.index <split_date]
x_train=train_data.drop('R',axis=1)
y_train=train_data[['R']]
test_data=df[df.index >=split_date]
x_test=test_data.drop('R',axis=1)
y_test=test_data[['R']]
x_train.shape
x_test.shape

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ### Create a new feature for B/M and DP

# %%

# 2. Perform feature engineering on the training set
x_train['dp_dy_ratio'] = (x_train['dp'] / x_train['dy'])
x_train = x_train.drop('dp', axis=1)
x_train = x_train.drop('dy', axis=1)

# 3. Apply the same feature engineering to the testing set
x_test['dp_dy_ratio'] = (x_test['dp'] / x_test['dy'])
x_test = x_test.drop('dp', axis=1)
x_test = x_test.drop('dy', axis=1)

# %% [markdown]
# #### Analyse New Features 

# %% [markdown]
# ### Moving Averages and Rolling Volitility 

# %% [markdown]
# #### Moving average for default yield spread

# %%
x_train['dfy_ma3'] = x_train['dfy'].rolling(window=3).mean()

x_test['dfy_ma3'] = x_test['dfy'].rolling(window=3).mean()



# %% [markdown]
# #### Moving average and rolling volatility for stock variance

# %%
x_train['svar_ma3'] = x_train['svar'].rolling(window=3).mean()

x_test['svar_ma3'] = x_test['svar'].rolling(window=3).mean()


# %% [markdown]
# #### Rolling volatility for inflation

# %%
x_train['infl_rolling_std'] = df['infl'].rolling(window=6).std()
x_test['infl_rolling_std'] = df['infl'].rolling(window=6).std()

# %% [markdown]
# #### Re-analyse data moving average and roling volitility 

# %%
plt.figure(figsize=(12, 10))
sns.heatmap(x_test.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# %% [markdown]
# ### Lagging 

# %% [markdown]
# #### Lag Market Returns 

# %%

lags_mr = list(range(1, 4,1)) + list(range(6, 13, 6))
# Apply lags for 'mr'
for lag in lags_mr:
    x_train[f'mr_lag{lag}'] = x_train['mr'].shift(lag)
    x_test[f'mr_lag{lag}'] = x_test['mr'].shift(lag)

# Handle missing values
x_train = x_train.fillna(method='bfill')
x_test = x_test.fillna(method='bfill')
x_train.dropna(inplace=True)

x_test.dropna(inplace=True)

# %% [markdown]
# #### Lag Fundamental fators 

# %%
lags_tms_dfy = list(range(1, 7, 2) ) # Monthly lags up to 12 months
for lag in lags_tms_dfy:
    x_train[f'tms_lag{lag}'] = x_train['tms'].shift(lag)
    x_test[f'tms_lag{lag}'] = x_test['tms'].shift(lag)
    x_train[f'ntis_lag{lag}'] = x_train['ntis'].shift(lag)
    x_test[f'ntis_lag{lag}'] = x_test['ntis'].shift(lag)
    x_train[f'dfy_ma3_lag{lag}'] = x_train['dfy_ma3'].shift(lag)
    x_test[f'dfy_ma3_lag{lag}'] = x_test['dfy_ma3'].shift(lag)

# Handle missing values
x_train = x_train.fillna(method='bfill')
x_test = x_test.fillna(method='bfill')
x_train.dropna(inplace=True)

x_test.dropna(inplace=True)


# %% [markdown]
# #### Lag Economic Factors

# %%
lags_infl_ltr_bm = list(range(1, 7,2))  # Monthly lags for slower-moving variables
# Apply lags for 'infl', 'ltr', and 'b/m'
for lag in lags_infl_ltr_bm:
    x_train[f'infl_lag{lag}'] = x_train['infl'].shift(lag)
    x_test[f'infl_lag{lag}'] = x_test['infl'].shift(lag)
    x_train[f'ltr_lag{lag}'] = x_train['ltr'].shift(lag)
    x_test[f'ltr_lag{lag}'] = x_test['ltr'].shift(lag)
    x_train[f'b/m{1}'] = x_train['b/m'].shift(1)
    x_test[f'b/m{1}'] = x_test['b/m'].shift(1)
# Handle missing values
x_train = x_train.fillna(method='bfill')
x_test = x_test.fillna(method='bfill')
x_train.dropna(inplace=True)

x_test.dropna(inplace=True)

 

# %%
# x_train['interaction_1'] = x_train['dfy_ma3_lag3'] * x_train['dp_dy_ratio']
# x_train['rolling_mean_mr'] = x_train['mr'].rolling(window=3).mean()
# x_train['volatility_normalized_dfy'] = x_train['dfy'] / x_train['svar']

# x_test['interaction_1'] = x_test['dfy_ma3_lag3'] * x_test['dp_dy_ratio'] 
# x_test['rolling_mean_mr'] = x_test['mr'].rolling(window=3).mean()
# x_test['volatility_normalized_dfy'] = x_test['dfy'] / x_test['svar']


# %%
x_train.describe()
x_test.describe()

# %% [markdown]
# #### Analyse

# %% [markdown]
# ### Dropping columns

# %% [markdown]
# #### Drop Book to market ratio

# %%
x_train = x_train.drop('b/m', axis=1) 
x_test = x_test.drop('b/m', axis=1)

# %% [markdown]
# #### Drop long term yield 
# Dropping LTY and TBL because the relationship is covered in term spread 

# %%
x_train = x_train.drop('lty', axis=1) 
x_test = x_test.drop('lty', axis=1)

# %%
x_test.describe()

# %% [markdown]
# ## Data Transformation
# The evaluation of the data revealed a high standard deviation in certain features. To mitigate the potential impact of this variability and ensure features contribute equally to model training, data standardization was applied. This process transforms the data to have zero mean and unit variance, effectively balancing the dataset.

# %% [markdown]
# ### Data Scaling

# %%
scaler_x = StandardScaler()
columns = x_train.columns

x_train[columns] = scaler_x.fit_transform(x_train[columns])  # Fit on x_train, transform x_train
x_test[columns] = scaler_x.transform(x_test[columns])        # Transform x_test using the same scaler

# %% [markdown]
# ### Analyse Transformation

# %%
x_test.describe()

# %% [markdown]
# # Models

# %% [markdown]
# ## Linear Regression

# %% [markdown]
# ### OLS

# %% [markdown]
# #### Training

# %%
ols = LinearRegression()
ols = ols.fit(x_train, y_train)
y_insample_pred_ols = ols.predict(x_train)
y_outsample_pred_ols = ols.predict(x_test)

# %% [markdown]
# #### Analysing OLS Coeeficient 

# %%
# Assuming 'ols' is the fitted Linear Regression model from your code
# Access coefficients
coefficients = ols.coef_

# Access feature names
feature_names = x_train.columns

# Create a DataFrame for coefficients and their importance
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients[0]})

# Sort by absolute value of coefficients (importance)
coefficients_df['Abs_Coefficient'] = np.abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)
coefficients_df = coefficients_df.drop(columns=['Abs_Coefficient'])

# Display the coefficients and their importance
coefficients_df

# %% [markdown]
# ### Ridge

# %% [markdown]
# #### Time serires Cross validation

# %%
tscv = TimeSeriesSplit(n_splits=5, test_size=12 )

# %% [markdown]
# ####  Alpha Cross Validation

# %%
alphas = np.logspace(-4, 4, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring='neg_mean_squared_error')
ridge_cv.fit(x_train, y_train)

# %% [markdown]
# #### Training

# %%
ridge = Ridge(alpha=ridge_cv.alpha_,max_iter=1000,fit_intercept=False)
ridge.fit(x_train, y_train)
y_insample_pred_ridge = ridge.predict(x_train)
y_outsample_pred_ridge = ridge.predict(x_test)

# %% [markdown]
# ### Lasso

# %% [markdown]
# #### Time serires Cross validation

# %%
tscv = TimeSeriesSplit(n_splits=5, test_size=12)

# %% [markdown]
# #### Alpha Cross Validation

# %%
alphas = np.logspace(-4, 4, 100) 
lasso_cv = LassoCV(alphas=alphas, cv=tscv)
lasso_cv.fit(x_train, y_train)

# %% [markdown]
# #### Analysing the coefficent

# %%
# @ Analysing the coefficent
coefficients = lasso_cv.coef_

feature_names = x_train.columns
dropped_features = feature_names[np.where(coefficients ==0)]
print(dropped_features)
import numpy as np

importance = np.abs(coefficients)
sorted_indices = np.argsort(importance)[::-1]  # Indices sorted by importance

# If you have feature names (e.g., from a pandas DataFrame):
for i in sorted_indices:
    print(f"{feature_names[i]}: {importance[i]}")

# %% [markdown]
# #### Training

# %%
lasso = Lasso(alpha=lasso_cv.alpha_,max_iter=1000,fit_intercept=False)
lasso.fit(x_train, y_train)
y_insample_pred_lasso = lasso.predict(x_train)
y_outsample_pred_lasso = lasso.predict(x_test)

# %% [markdown]
# ## Random Forest

# %%

# Implement Random Forest with optimized hyperparameter tuning
rf = RandomForestRegressor(random_state=42, n_jobs=- 4)  # Use all CPU cores

rf_params = {
    "n_estimators": [50, 100, 200, 300],  # Fewer estimators for faster training
    "max_depth": [None, 10, 20, 30],     # Balanced depth options
    "min_samples_split": [2, 5, 10, 15], # Tuning sample split criteria
}

# Use fewer CV folds if speed is critical
rf_random = RandomizedSearchCV(
    rf, rf_params, cv=TimeSeriesSplit(n_splits=4, test_size=12), scoring="neg_mean_squared_error"
)


# Ensure y_train is a 1D array
y_train_1d = y_train.to_numpy().ravel()

# Fit the model
rf_random.fit(x_train, y_train_1d)

print("Best Random Forest Params:", rf_random.best_params_)

# Predict and evaluate Random Forest
y_train_pred_rf = rf_random.best_estimator_.predict(x_train)
y_test_pred_rf = rf_random.best_estimator_.predict(x_test)

# %% [markdown]
# # Performance Test

# %% [markdown]
# ## Performance Metrics

# %% [markdown]
# ### Define a function to calculate the performance metrics

# %%
import matplotlib.pyplot as plt
import pandas as pd

def timing_strategy_evaluation_with_drawdown(trained_model, X_test, actual_returns, risk_free_rate=0.02 / 12, threshold=0, initial_value=100):
    """
    Evaluate a timing strategy based on a trained model's predictions.

    Parameters:
        trained_model: Trained machine learning model with a `predict` method.
        X_test: DataFrame or array of predictors for testing (features for prediction).
        actual_returns: Series or array of actual returns for the evaluation period.
        risk_free_rate: Monthly risk-free rate, default is 0.02 annualized.
        threshold: Threshold for deciding risk-on or risk-off, default is 0.
        initial_value: Initial portfolio value, default is 100.

    Returns:
        portfolio_values: Series of portfolio values over time.
        cumulative_return: Final cumulative return of the portfolio.
        sharpe_ratio: Sharpe ratio of the portfolio strategy.
        max_drawdown: Maximum drawdown of the portfolio.
    """
    # Predict returns using the trained model
    predicted_returns = trained_model.predict(X_test)
    
    # Ensure actual_returns is a NumPy array for consistency
    if isinstance(actual_returns, pd.Series) or isinstance(actual_returns, pd.DataFrame):
        actual_returns = actual_returns.values.flatten()
    elif not isinstance(actual_returns, (list, tuple)):
        raise TypeError("actual_returns must be a Series, DataFrame, list, or tuple.")

    # Initialize portfolio for timing strategy
    portfolio_values_timing = [initial_value]

    # Timing strategy
    for i in range(len(predicted_returns)):
        if predicted_returns[i] > threshold:  # Risk-On
            portfolio_values_timing.append(portfolio_values_timing[-1] * (1 + actual_returns[i]))
        else:  # Risk-Off
            portfolio_values_timing.append(portfolio_values_timing[-1] * (1 + risk_free_rate))

    # Convert portfolio values to pandas Series for analysis
    portfolio_values_timing = pd.Series(portfolio_values_timing)

    # Calculate performance metrics for timing strategy
    cumulative_return_timing = portfolio_values_timing.iloc[-1] / portfolio_values_timing.iloc[0] - 1
    sharpe_ratio_timing = (portfolio_values_timing.pct_change().mean() - risk_free_rate) / portfolio_values_timing.pct_change().std()

    # Calculate maximum drawdown for timing strategy
    rolling_max_timing = portfolio_values_timing.cummax()
    drawdown_timing = (portfolio_values_timing - rolling_max_timing) / rolling_max_timing
    max_drawdown_timing = drawdown_timing.min()

    # Buy-and-hold strategy
    portfolio_values_bh = [initial_value]
    for ret in actual_returns:
        portfolio_values_bh.append(portfolio_values_bh[-1] * (1 + ret))
    portfolio_values_bh = pd.Series(portfolio_values_bh)

    # Plot portfolio evolution
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values_timing, label="Timing Strategy", marker='o', linestyle='-')
    plt.plot(portfolio_values_bh, label="Buy-and-Hold Strategy", marker='x', linestyle='--')
    plt.title("Portfolio Evolution: Timing Strategy vs Buy-and-Hold")
    plt.xlabel("Time (Months)")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()

    return portfolio_values_timing, cumulative_return_timing, sharpe_ratio_timing, max_drawdown_timing



# %% [markdown]
# ### Linear Regression 

# %% [markdown]
# #### In-sample Performance Comparism

# %%
mse_insample_ols = mean_squared_error(y_train, y_insample_pred_ols)
r2_insample_ols = r2_score(y_train, y_insample_pred_ols)
mse_insample_ridge = mean_squared_error(y_train, y_insample_pred_ridge)
r2_insample_ridge = r2_score(y_train, y_insample_pred_ridge)
mse_insample_lasso = mean_squared_error(y_train, y_insample_pred_lasso)
r2_insample_lasso = r2_score(y_train, y_insample_pred_lasso)
print('Model Insample Performance Comparison:')
print(f'OLS MSE: {mse_insample_ols:.4f}, R-squared: {r2_insample_ols:.4f}')
print(f'Ridge MSE: {mse_insample_ridge:.4f}, R-squared: {r2_insample_ridge:.4f}')
print(f'Lasso MSE: {mse_insample_lasso:.4f}, R-squared: {r2_insample_lasso:.4f}')


# %% [markdown]
# #### out-sample Performance Comparism

# %%
mse_outsample_ols = mean_squared_error(y_test, y_outsample_pred_ols)
r2_outsample_ols = r2_score(y_test, y_outsample_pred_ols)
mse_outsample_ridge = mean_squared_error(y_test, y_outsample_pred_ridge)
r2_outsample_ridge = r2_score(y_test, y_outsample_pred_ridge)
mse_outsample_lasso = mean_squared_error(y_test, y_outsample_pred_lasso)
r2_outsample_lasso = r2_score(y_test, y_outsample_pred_lasso)
print('Model Outsample Performance Comparison:')
print(f'OLS MSE: {mse_outsample_ols:.4f}, R-squared: {r2_outsample_ols:.4f}')
print(f'Ridge MSE: {mse_outsample_ridge:.4f}, R-squared: {r2_outsample_ridge:.4f}')
print(f'Lasso MSE: {mse_outsample_lasso:.4f}, R-squared: {r2_outsample_lasso:.4f}')

# %% [markdown]
# ### Timing Strategy for OLS
# 

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(ols, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ### Timing Strategy for Ridge

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(ridge, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ### Timing Strategy for Lasso
# 

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(lasso, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ### Random Forest

# %% [markdown]
# #### insample Performance comparism 

# %%
mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
r2_train_rf = r2_score(y_train, y_train_pred_rf)
print(f"Random Forest - Training MSE: {mse_train_rf:.4f}, Training R²: {r2_train_rf:.4f}")

# %% [markdown]
# #### out-sample Performance comparism

# %%
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print(f"Random Forest - Testing MSE: {mse_test_rf:.4f}, Testing R²: {r2_test_rf:.4f}")

# %% [markdown]
# #### Timing Strategy for Random Forest
# 

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(rf_random, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ## Visuals

# %% [markdown]
# ### Plot for Linear Regression   
# 

# %%
# In-sample Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_train.index, y_train, label="Actual (Train)", color="blue")
plt.plot(y_train.index, y_insample_pred_ols, label="Predicted (Train)", color="orange")
plt.title("In-sample Predictions")
plt.xlabel("Date")
plt.ylabel("R")
plt.legend()
plt.show()

# Out-of-sample Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual (Test)", color="blue")
plt.plot(y_test.index, y_outsample_pred_ols, label="Predicted (Test)", color="green")
plt.title("Out-of-sample Predictions")
plt.xlabel("Date")
plt.ylabel("R")
plt.legend()
plt.show()


# %% [markdown]
# ### Plot for Random Forest
# 

# %%
plt.figure(figsize=(12, 6))

# In-sample plot
plt.subplot(1, 2, 1)
plt.plot(y_train.index, y_train, label='Actual', color='blue')
plt.plot(y_train.index, y_train_pred_rf, label='Predicted', color='red')
plt.title('Random Forest - In-sample Predictions')
plt.xlabel('Date')
plt.ylabel('R')
plt.legend()

# Out-of-sample plot
plt.subplot(1, 2, 2)
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_test_pred_rf, label='Predicted', color='green')
plt.title('Random Forest - Out-of-sample Predictions')
plt.xlabel('Date')
plt.ylabel('R')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# # Financial Analysis

# %% [markdown]
# ## Analysing data of Know Historical events

# %%


# %% [markdown]
# ## Market Valuation Signal

# %% [markdown]
# ### Dividend to Price Ratio vs. Book to Market Ratio
# The Graph shows strong correlation between the book to market ratio and Divends to price ratio

# %%
plt.figure(figsize=(8, 6))
plt.scatter(DF_FINAL['dp'], DF_FINAL['b/m'], c=DF_FINAL['tms'], cmap='viridis')
plt.xlabel('Dividend to Price Ratio (dp)')
plt.ylabel('Book to Market Ratio (b/m)')
plt.title('Dividend to Price Ratio vs. Book to Market Ratio')
_ = plt.colorbar(label='Market Risk Premium (tms)')

# %% [markdown]
# ### Dividend price vs Dividend Yield
# Dividend price vs Dividend Yield

# %%
plt.figure(figsize=(8, 6))
plt.scatter(DF_FINAL['dp'], DF_FINAL['dy'], c=df['tms'], cmap='viridis')
plt.xlabel('Dividend to Price Ratio (dp)')
plt.ylabel('Dividend to Yields (d/y)')
plt.title('Dividend to Price Ratio vs. Dividend to Yields ')
_ = plt.colorbar(label='Market Risk Premium (tms)')

# %% [markdown]
# ##  Spike Analysis 

# %%
# Assuming 'R' column represents returns and the index is a datetime index.
def find_spike_periods(df, return_column='R', threshold=2):
    """
    Finds periods of spikes in returns exceeding a given threshold.

    Args:
        DF_FINAL: DataFrame with a datetime index and a return column.
        return_column: The name of the column containing returns.
        threshold: The standard deviation threshold to identify a spike.

    Returns:
        A list of tuples, where each tuple represents a spike period
        (start_date, end_date).
    """

    # Calculate rolling standard deviation to identify volatility
    rolling_std = DF_FINAL[return_column].rolling(window=12).std() # Adjust window size as needed

    # Identify spikes based on threshold
    spikes = DF_FINAL[return_column][rolling_std > threshold * rolling_std.mean()]

    # Group consecutive spikes into periods
    spike_periods = []
    start_date = None
    for date in spikes.index:
        if start_date is None:
            start_date = date
        elif date != spikes.index[spikes.index.get_loc(date) - 1] + pd.DateOffset(months=1): # Adjust for your data freq
            spike_periods.append((start_date, spikes.index[spikes.index.get_loc(date) - 1]))
            start_date = date
    if start_date is not None:
        spike_periods.append((start_date, spikes.index[-1]))

    return spike_periods

# Example usage:
spike_periods = find_spike_periods(DF_FINAL)
print(spike_periods)

# For visualization
plt.figure(figsize=(12, 6))
plt.plot(DF_FINAL['R'], label='Returns')
plt.plot(DF_FINAL['R'].rolling(window=6).std(), label='Rolling Std Dev')

for start, end in spike_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Spike Period' if start==spike_periods[0][0] else '') # Plot each spike as a shaded area
plt.legend()
plt.title('Return Spikes')
plt.xlabel('Date')
plt.ylabel('Return')
plt.show()

# %% [markdown]
# # Pre-requisite
# 
# 
# 

# %% [markdown]
# ## Import Libraries
# 

# %%
# Comment out the pip requirement 
# !pip install -r requirements.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Final
import sklearn
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_ccf
from statsmodels.tsa.stattools import acf, pacf
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# %% [markdown]
# ## Load The Data

# %%
df = pd.read_csv('https://raw.githubusercontent.com/demilade27/Predicting-Monthly-Excess-Returns-of-Market-Index/d2f2cb8478612fa4e8fd4e87628375d44f6cb72e/data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
DF_FINAL: Final = df 

# %% [markdown]
# # Analyse the dataset

# %% [markdown]
# ## Generate descriptive statistics

# %%
df.describe()

# %% [markdown]
# ## Check for null or zero values
# *Analysis*
# ---
# There are no null values in the dataset
# Analysing the dataset values there are zero values
# * INFL: There are 239 zero values showing signs of Deflationary Stagnation
# * DE:
# * LTR:
# * TMS:
# * DFR:

# %%
print(df.isnull().sum())
print(df.duplicated().sum())
print((df == 0).sum())

# %% [markdown]
# ## Analyse Data Skewness

# %%
df.skew()

# %% [markdown]
# 
# ## Data Visualization
# 

# %% [markdown]
# ### Correlation analysis

# %% [markdown]
# #### Correlation analysis of dataset

# %%
correlation_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,fmt=".2f",cmap="coolwarm")
plt.title('Correlation Heatmap')

# %% [markdown]
# #### Correlation analysis with R

# %%
target_correlation = correlation_matrix[['R']]
plt.figure(figsize=(10,8))
sns.heatmap(target_correlation,annot=True,fmt=".2f",cmap="coolwarm")
plt.title('Correlation Heatmap with R')

# %% [markdown]
# ### Residual plot
# 

# %%
sns.pairplot(df,y_vars=['R'],x_vars=df.select_dtypes(include='number').columns,kind='reg')
plt.suptitle("Pairwise Scatterplots with Fitted Lines")
plt.show()

# %% [markdown]
# ### Box Plot
# This is the analysis of the skewness of the data

# %%
plt.figure(figsize=(12, 6))
df.boxplot()
plt.title('Box Plot of Training Data Features')
plt.xticks(rotation=45, ha='right')
plt.show()

# %% [markdown]
# ### Autocorrelation and Partial Autocorrelation Analysis

# %% [markdown]
# #### 12 Month lag

# %%
plot_acf(df['mr'],lags=12)
plot_pacf(df['mr'],lags=12)
plt.show()

# %% [markdown]
# #### 24 Month lag

# %%
plot_acf(df['mr'],lags=24)
plot_pacf(df['mr'],lags=24)
plt.show()

# %% [markdown]
# # Pre-processing

# %% [markdown]
# ## Data Spliting

# %%
split_date='2019-01-01'
train_data=df[df.index <split_date]
x_train=train_data.drop('R',axis=1)
y_train=train_data[['R']]
test_data=df[df.index >=split_date]
x_test=test_data.drop('R',axis=1)
y_test=test_data[['R']]
x_train.shape
x_test.shape

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ### Create a new feature for B/M and DP

# %%
interaction_pairs = [('tms', 'infl'), ('dp', 'dy'), ('dfy', 'tbl')]
for var1, var2 in interaction_pairs:
    x_train[f'{var1}_x_{var2}'] = x_train[var1] * x_train[var2]
    x_test[f'{var1}_x_{var2}'] = x_test[var1] * x_test[var2]

# %% [markdown]
# #### Analyse New Features 

# %% [markdown]
# ### Moving Averages and Rolling Volitility 

# %% [markdown]
# #### Moving average for default yield spread

# %%
x_train['dfy_ma3'] = x_train['dfy'].rolling(window=3).mean()

x_test['dfy_ma3'] = x_test['dfy'].rolling(window=3).mean()



# %% [markdown]
# #### Moving average and rolling volatility for stock variance

# %%
x_train['svar_ma3'] = x_train['svar'].rolling(window=3).mean()

x_test['svar_ma3'] = x_test['svar'].rolling(window=3).mean()


# %% [markdown]
# #### Rolling volatility for inflation

# %%
x_train['infl_rolling_std'] = df['infl'].rolling(window=6).std()
x_test['infl_rolling_std'] = df['infl'].rolling(window=6).std()

# %% [markdown]
# #### Re-analyse data moving average and roling volitility 

# %%
plt.figure(figsize=(12, 10))
sns.heatmap(x_test.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# %% [markdown]
# ### Lagging 

# %% [markdown]
# #### Lag Market Returns 

# %% [markdown]
# <a href="https://colab.research.google.com/github/demilade27/Predicting-Monthly-Excess-Returns-of-Market-Index/blob/main/Predicting_Monthly_Excess_Returns_of_Market_Index.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Pre-requisite
# 
# 
# 

# %% [markdown]
# ## Import Libraries
# 

# %%
# Comment out the pip requirement 
# !pip install -r requirements.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Final
import sklearn
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_ccf
from statsmodels.tsa.stattools import acf, pacf
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# %% [markdown]
# ## Load The Data

# %%
df = pd.read_csv('https://raw.githubusercontent.com/demilade27/Predicting-Monthly-Excess-Returns-of-Market-Index/d2f2cb8478612fa4e8fd4e87628375d44f6cb72e/data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
DF_FINAL: Final = df 

# %% [markdown]
# # Analyse the dataset

# %% [markdown]
# ## Generate descriptive statistics

# %%
df.describe()

# %% [markdown]
# ## Check for null or zero values
# *Analysis*
# ---
# There are no null values in the dataset
# Analysing the dataset values there are zero values
# * INFL: There are 239 zero values showing signs of Deflationary Stagnation
# * DE:
# * LTR:
# * TMS:
# * DFR:

# %%
print(df.isnull().sum())
print(df.duplicated().sum())
print((df == 0).sum())

# %% [markdown]
# ## Analyse Data Skewness

# %%
df.skew()

# %% [markdown]
# 
# ## Data Visualization
# 

# %% [markdown]
# ### Correlation analysis

# %% [markdown]
# #### Correlation analysis of dataset

# %%
correlation_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,fmt=".2f",cmap="coolwarm")
plt.title('Correlation Heatmap')

# %% [markdown]
# #### Correlation analysis with R

# %%
target_correlation = correlation_matrix[['R']]
plt.figure(figsize=(10,8))
sns.heatmap(target_correlation,annot=True,fmt=".2f",cmap="coolwarm")
plt.title('Correlation Heatmap with R')

# %% [markdown]
# ### Residual plot
# 

# %%
sns.pairplot(df,y_vars=['R'],x_vars=df.select_dtypes(include='number').columns,kind='reg')
plt.suptitle("Pairwise Scatterplots with Fitted Lines")
plt.show()

# %% [markdown]
# ### Box Plot
# This is the analysis of the skewness of the data

# %%
plt.figure(figsize=(12, 6))
df.boxplot()
plt.title('Box Plot of Training Data Features')
plt.xticks(rotation=45, ha='right')
plt.show()

# %% [markdown]
# ### Autocorrelation and Partial Autocorrelation Analysis

# %% [markdown]
# #### 12 Month lag

# %%
plot_acf(df['mr'],lags=12)
plot_pacf(df['mr'],lags=12)
plt.show()

# %% [markdown]
# #### 24 Month lag

# %%
plot_acf(df['mr'],lags=24)
plot_pacf(df['mr'],lags=24)
plt.show()

# %% [markdown]
# # Pre-processing

# %% [markdown]
# ## Data Spliting

# %%
split_date='2019-01-01'
train_data=df[df.index <split_date]
x_train=train_data.drop('R',axis=1)
y_train=train_data[['R']]
test_data=df[df.index >=split_date]
x_test=test_data.drop('R',axis=1)
y_test=test_data[['R']]
x_train.shape
x_test.shape

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ### Create a new feature for B/M and DP

# %%
interaction_pairs = [('tms', 'infl'), ('dp', 'dy'), ('dfy', 'tbl')]
for var1, var2 in interaction_pairs:
    x_train[f'{var1}_x_{var2}'] = x_train[var1] * x_train[var2]
    x_test[f'{var1}_x_{var2}'] = x_test[var1] * x_test[var2]

# %% [markdown]
# #### Analyse New Features 

# %% [markdown]
# ### Moving Averages and Rolling Volitility 

# %% [markdown]
# #### Moving average for default yield spread

# %%
x_train['dfy_ma3'] = x_train['dfy'].rolling(window=3).mean()

x_test['dfy_ma3'] = x_test['dfy'].rolling(window=3).mean()



# %% [markdown]
# #### Moving average and rolling volatility for stock variance

# %%
x_train['svar_ma3'] = x_train['svar'].rolling(window=3).mean()

x_test['svar_ma3'] = x_test['svar'].rolling(window=3).mean()


# %% [markdown]
# #### Rolling volatility for inflation

# %%
x_train['infl_rolling_std'] = df['infl'].rolling(window=6).std()
x_test['infl_rolling_std'] = df['infl'].rolling(window=6).std()

# %% [markdown]
# #### Re-analyse data moving average and roling volitility 

# %%
plt.figure(figsize=(12, 10))
sns.heatmap(x_test.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# %%
for var in ['dp', 'dy', 'ep', 'b/m']:
    mean_val = train_data[var].mean()  # Calculate mean from training data
    x_train[f'{var}_normalized'] = x_train[var] / mean_val
    x_test[f'{var}_normalized'] = x_test[var] / mean_val

# %% [markdown]
# ### Lagging 

# %% [markdown]
# #### Lag Market Returns 

# %%

lags_mr = list(range(1, 4,1)) + list(range(6, 13, 6))
# Apply lags for 'mr'
for lag in lags_mr:
    x_train[f'mr_lag{lag}'] = x_train['mr'].shift(lag)
    x_test[f'mr_lag{lag}'] = x_test['mr'].shift(lag)

# Handle missing values
x_train = x_train.fillna(method='bfill')
x_test = x_test.fillna(method='bfill')
x_train.dropna(inplace=True)

x_test.dropna(inplace=True)

# %% [markdown]
# #### Lag Fundamental fators 

# %%
lags_tms_dfy = list(range(1, 7, 2) ) # Monthly lags up to 12 months
for lag in lags_tms_dfy:
    x_train[f'tms_lag{lag}'] = x_train['tms'].shift(lag)
    x_test[f'tms_lag{lag}'] = x_test['tms'].shift(lag)
    x_train[f'ntis_lag{lag}'] = x_train['ntis'].shift(lag)
    x_test[f'ntis_lag{lag}'] = x_test['ntis'].shift(lag)
    x_train[f'dfy_ma3_lag{lag}'] = x_train['dfy_ma3'].shift(lag)
    x_test[f'dfy_ma3_lag{lag}'] = x_test['dfy_ma3'].shift(lag)

# Handle missing values
x_train = x_train.fillna(method='bfill')
x_test = x_test.fillna(method='bfill')
x_train.dropna(inplace=True)

x_test.dropna(inplace=True)


# %% [markdown]
# #### Lag Economic Factors

# %%
lags_infl_ltr_bm = list(range(1, 7,2))  # Monthly lags for slower-moving variables
# Apply lags for 'infl', 'ltr', and 'b/m'
for lag in lags_infl_ltr_bm:
    x_train[f'infl_lag{lag}'] = x_train['infl'].shift(lag)
    x_test[f'infl_lag{lag}'] = x_test['infl'].shift(lag)
    x_train[f'ltr_lag{lag}'] = x_train['ltr'].shift(lag)
    x_test[f'ltr_lag{lag}'] = x_test['ltr'].shift(lag)
    x_train[f'b/m{1}'] = x_train['b/m'].shift(1)
    x_test[f'b/m{1}'] = x_test['b/m'].shift(1)
# Handle missing values
x_train = x_train.fillna(method='bfill')
x_test = x_test.fillna(method='bfill')
x_train.dropna(inplace=True)

x_test.dropna(inplace=True)

 

# %%
# x_train['interaction_1'] = x_train['dfy_ma3_lag3'] * x_train['dp_dy_ratio']
# x_train['rolling_mean_mr'] = x_train['mr'].rolling(window=3).mean()
# x_train['volatility_normalized_dfy'] = x_train['dfy'] / x_train['svar']

# x_test['interaction_1'] = x_test['dfy_ma3_lag3'] * x_test['dp_dy_ratio'] 
# x_test['rolling_mean_mr'] = x_test['mr'].rolling(window=3).mean()
# x_test['volatility_normalized_dfy'] = x_test['dfy'] / x_test['svar']


# %%
x_train.describe()
x_test.describe()

# %% [markdown]
# #### Analyse

# %% [markdown]
# ### Dropping columns

# %% [markdown]
# #### Drop Book to market ratio

# %%
x_train = x_train.drop('b/m', axis=1) 
x_test = x_test.drop('b/m', axis=1)

# %% [markdown]
# #### Drop long term yield 
# Dropping LTY and TBL because the relationship is covered in term spread 

# %%
x_train = x_train.drop('lty', axis=1) 
x_test = x_test.drop('lty', axis=1)

# %%
x_test.describe()

# %% [markdown]
# ## Data Transformation
# The evaluation of the data revealed a high standard deviation in certain features. To mitigate the potential impact of this variability and ensure features contribute equally to model training, data standardization was applied. This process transforms the data to have zero mean and unit variance, effectively balancing the dataset.

# %% [markdown]
# ### Data Scaling

# %%
scaler_x = StandardScaler()
columns = x_train.columns

x_train[columns] = scaler_x.fit_transform(x_train[columns])  # Fit on x_train, transform x_train
x_test[columns] = scaler_x.transform(x_test[columns])        # Transform x_test using the same scaler

# %% [markdown]
# ### Analyse Transformation

# %%
x_test.describe()

# %% [markdown]
# # Models

# %% [markdown]
# ## Linear Regression

# %% [markdown]
# ### OLS

# %% [markdown]
# #### Training

# %%
ols = LinearRegression()
ols = ols.fit(x_train, y_train)
y_insample_pred_ols = ols.predict(x_train)
y_outsample_pred_ols = ols.predict(x_test)

# %% [markdown]
# #### Analysing OLS Coeeficient 

# %%
# Assuming 'ols' is the fitted Linear Regression model from your code
# Access coefficients
coefficients = ols.coef_

# Access feature names
feature_names = x_train.columns

# Create a DataFrame for coefficients and their importance
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients[0]})

# Sort by absolute value of coefficients (importance)
coefficients_df['Abs_Coefficient'] = np.abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)
coefficients_df = coefficients_df.drop(columns=['Abs_Coefficient'])

# Display the coefficients and their importance
coefficients_df

# %% [markdown]
# ### Ridge

# %% [markdown]
# #### Time serires Cross validation

# %%
tscv = TimeSeriesSplit(n_splits=5, test_size=12 )

# %% [markdown]
# ####  Alpha Cross Validation

# %%
alphas = np.logspace(-4, 4, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring='neg_mean_squared_error')
ridge_cv.fit(x_train, y_train)

# %% [markdown]
# #### Training

# %%
ridge = Ridge(alpha=ridge_cv.alpha_,max_iter=1000,fit_intercept=False)
ridge.fit(x_train, y_train)
y_insample_pred_ridge = ridge.predict(x_train)
y_outsample_pred_ridge = ridge.predict(x_test)

# %% [markdown]
# ### Lasso

# %% [markdown]
# #### Time serires Cross validation

# %%
tscv = TimeSeriesSplit(n_splits=5, test_size=12)

# %% [markdown]
# #### Alpha Cross Validation

# %%
alphas = np.logspace(-4, 4, 100) 
lasso_cv = LassoCV(alphas=alphas, cv=tscv)
lasso_cv.fit(x_train, y_train)

# %% [markdown]
# #### Analysing the coefficent

# %%
# @ Analysing the coefficent
coefficients = lasso_cv.coef_

feature_names = x_train.columns
dropped_features = feature_names[np.where(coefficients ==0)]
print(dropped_features)
import numpy as np

importance = np.abs(coefficients)
sorted_indices = np.argsort(importance)[::-1]  # Indices sorted by importance

# If you have feature names (e.g., from a pandas DataFrame):
for i in sorted_indices:
    print(f"{feature_names[i]}: {importance[i]}")

# %% [markdown]
# #### Training

# %%
lasso = Lasso(alpha=lasso_cv.alpha_,max_iter=1000,fit_intercept=False)
lasso.fit(x_train, y_train)
y_insample_pred_lasso = lasso.predict(x_train)
y_outsample_pred_lasso = lasso.predict(x_test)

# %% [markdown]
# ## Random Forest

# %%

# Implement Random Forest with optimized hyperparameter tuning
rf = RandomForestRegressor(random_state=42, n_jobs=- 4)  # Use all CPU cores

rf_params = {
    "n_estimators": [50, 100, 200, 300],  # Fewer estimators for faster training
    "max_depth": [None, 10, 20, 30],     # Balanced depth options
    "min_samples_split": [2, 5, 10, 15], # Tuning sample split criteria
}

# Use fewer CV folds if speed is critical
rf_random = RandomizedSearchCV(
    rf, rf_params, cv=TimeSeriesSplit(n_splits=4, test_size=12), scoring="neg_mean_squared_error"
)


# Ensure y_train is a 1D array
y_train_1d = y_train.to_numpy().ravel()

# Fit the model
rf_random.fit(x_train, y_train_1d)

print("Best Random Forest Params:", rf_random.best_params_)

# Predict and evaluate Random Forest
y_train_pred_rf = rf_random.best_estimator_.predict(x_train)
y_test_pred_rf = rf_random.best_estimator_.predict(x_test)

# %% [markdown]
# # Performance Test

# %% [markdown]
# ## Performance Metrics

# %% [markdown]
# ### Define a function to calculate the performance metrics

# %%
import matplotlib.pyplot as plt
import pandas as pd

def timing_strategy_evaluation_with_drawdown(trained_model, X_test, actual_returns, risk_free_rate=0.02 / 12, threshold=0, initial_value=100):
    """
    Evaluate a timing strategy based on a trained model's predictions.

    Parameters:
        trained_model: Trained machine learning model with a `predict` method.
        X_test: DataFrame or array of predictors for testing (features for prediction).
        actual_returns: Series or array of actual returns for the evaluation period.
        risk_free_rate: Monthly risk-free rate, default is 0.02 annualized.
        threshold: Threshold for deciding risk-on or risk-off, default is 0.
        initial_value: Initial portfolio value, default is 100.

    Returns:
        portfolio_values: Series of portfolio values over time.
        cumulative_return: Final cumulative return of the portfolio.
        sharpe_ratio: Sharpe ratio of the portfolio strategy.
        max_drawdown: Maximum drawdown of the portfolio.
    """
    # Predict returns using the trained model
    predicted_returns = trained_model.predict(X_test)
    
    # Ensure actual_returns is a NumPy array for consistency
    if isinstance(actual_returns, pd.Series) or isinstance(actual_returns, pd.DataFrame):
        actual_returns = actual_returns.values.flatten()
    elif not isinstance(actual_returns, (list, tuple)):
        raise TypeError("actual_returns must be a Series, DataFrame, list, or tuple.")

    # Initialize portfolio for timing strategy
    portfolio_values_timing = [initial_value]

    # Timing strategy
    for i in range(len(predicted_returns)):
        if predicted_returns[i] > threshold:  # Risk-On
            portfolio_values_timing.append(portfolio_values_timing[-1] * (1 + actual_returns[i]))
        else:  # Risk-Off
            portfolio_values_timing.append(portfolio_values_timing[-1] * (1 + risk_free_rate))

    # Convert portfolio values to pandas Series for analysis
    portfolio_values_timing = pd.Series(portfolio_values_timing)

    # Calculate performance metrics for timing strategy
    cumulative_return_timing = portfolio_values_timing.iloc[-1] / portfolio_values_timing.iloc[0] - 1
    sharpe_ratio_timing = (portfolio_values_timing.pct_change().mean() - risk_free_rate) / portfolio_values_timing.pct_change().std()

    # Calculate maximum drawdown for timing strategy
    rolling_max_timing = portfolio_values_timing.cummax()
    drawdown_timing = (portfolio_values_timing - rolling_max_timing) / rolling_max_timing
    max_drawdown_timing = drawdown_timing.min()

    # Buy-and-hold strategy
    portfolio_values_bh = [initial_value]
    for ret in actual_returns:
        portfolio_values_bh.append(portfolio_values_bh[-1] * (1 + ret))
    portfolio_values_bh = pd.Series(portfolio_values_bh)

    # Plot portfolio evolution
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values_timing, label="Timing Strategy", marker='o', linestyle='-')
    plt.plot(portfolio_values_bh, label="Buy-and-Hold Strategy", marker='x', linestyle='--')
    plt.title("Portfolio Evolution: Timing Strategy vs Buy-and-Hold")
    plt.xlabel("Time (Months)")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()

    return portfolio_values_timing, cumulative_return_timing, sharpe_ratio_timing, max_drawdown_timing



# %% [markdown]
# ### Linear Regression 

# %% [markdown]
# #### In-sample Performance Comparism

# %%
mse_insample_ols = mean_squared_error(y_train, y_insample_pred_ols)
r2_insample_ols = r2_score(y_train, y_insample_pred_ols)
mse_insample_ridge = mean_squared_error(y_train, y_insample_pred_ridge)
r2_insample_ridge = r2_score(y_train, y_insample_pred_ridge)
mse_insample_lasso = mean_squared_error(y_train, y_insample_pred_lasso)
r2_insample_lasso = r2_score(y_train, y_insample_pred_lasso)
print('Model Insample Performance Comparison:')
print(f'OLS MSE: {mse_insample_ols:.4f}, R-squared: {r2_insample_ols:.4f}')
print(f'Ridge MSE: {mse_insample_ridge:.4f}, R-squared: {r2_insample_ridge:.4f}')
print(f'Lasso MSE: {mse_insample_lasso:.4f}, R-squared: {r2_insample_lasso:.4f}')


# %% [markdown]
# #### out-sample Performance Comparism

# %%
mse_outsample_ols = mean_squared_error(y_test, y_outsample_pred_ols)
r2_outsample_ols = r2_score(y_test, y_outsample_pred_ols)
mse_outsample_ridge = mean_squared_error(y_test, y_outsample_pred_ridge)
r2_outsample_ridge = r2_score(y_test, y_outsample_pred_ridge)
mse_outsample_lasso = mean_squared_error(y_test, y_outsample_pred_lasso)
r2_outsample_lasso = r2_score(y_test, y_outsample_pred_lasso)
print('Model Outsample Performance Comparison:')
print(f'OLS MSE: {mse_outsample_ols:.4f}, R-squared: {r2_outsample_ols:.4f}')
print(f'Ridge MSE: {mse_outsample_ridge:.4f}, R-squared: {r2_outsample_ridge:.4f}')
print(f'Lasso MSE: {mse_outsample_lasso:.4f}, R-squared: {r2_outsample_lasso:.4f}')

# %% [markdown]
# ### Timing Strategy for OLS
# 

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(ols, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ### Timing Strategy for Ridge

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(ridge, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ### Timing Strategy for Lasso
# 

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(lasso, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ### Random Forest

# %% [markdown]
# #### insample Performance comparism 

# %%
mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
r2_train_rf = r2_score(y_train, y_train_pred_rf)
print(f"Random Forest - Training MSE: {mse_train_rf:.4f}, Training R²: {r2_train_rf:.4f}")

# %% [markdown]
# #### out-sample Performance comparism

# %%
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print(f"Random Forest - Testing MSE: {mse_test_rf:.4f}, Testing R²: {r2_test_rf:.4f}")

# %% [markdown]
# #### Timing Strategy for Random Forest
# 

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(rf_random, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ## Visuals

# %% [markdown]
# ### Plot for Linear Regression   
# 

# %%
# In-sample Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_train.index, y_train, label="Actual (Train)", color="blue")
plt.plot(y_train.index, y_insample_pred_ols, label="Predicted (Train)", color="orange")
plt.title("In-sample Predictions")
plt.xlabel("Date")
plt.ylabel("R")
plt.legend()
plt.show()

# Out-of-sample Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual (Test)", color="blue")
plt.plot(y_test.index, y_outsample_pred_ols, label="Predicted (Test)", color="green")
plt.title("Out-of-sample Predictions")
plt.xlabel("Date")
plt.ylabel("R")
plt.legend()
plt.show()


# %% [markdown]
# ### Plot for Random Forest
# 

# %%
plt.figure(figsize=(12, 6))

# In-sample plot
plt.subplot(1, 2, 1)
plt.plot(y_train.index, y_train, label='Actual', color='blue')
plt.plot(y_train.index, y_train_pred_rf, label='Predicted', color='red')
plt.title('Random Forest - In-sample Predictions')
plt.xlabel('Date')
plt.ylabel('R')
plt.legend()

# Out-of-sample plot
plt.subplot(1, 2, 2)
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_test_pred_rf, label='Predicted', color='green')
plt.title('Random Forest - Out-of-sample Predictions')
plt.xlabel('Date')
plt.ylabel('R')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# # Financial Analysis

# %% [markdown]
# ## Analysing data of Know Historical events

# %%


# %% [markdown]
# ## Market Valuation Signal

# %% [markdown]
# ### Dividend to Price Ratio vs. Book to Market Ratio
# The Graph shows strong correlation between the book to market ratio and Divends to price ratio

# %%
plt.figure(figsize=(8, 6))
plt.scatter(DF_FINAL['dp'], DF_FINAL['b/m'], c=DF_FINAL['tms'], cmap='viridis')
plt.xlabel('Dividend to Price Ratio (dp)')
plt.ylabel('Book to Market Ratio (b/m)')
plt.title('Dividend to Price Ratio vs. Book to Market Ratio')
_ = plt.colorbar(label='Market Risk Premium (tms)')

# %% [markdown]
# ### Dividend price vs Dividend Yield
# Dividend price vs Dividend Yield

# %%
plt.figure(figsize=(8, 6))
plt.scatter(DF_FINAL['dp'], DF_FINAL['dy'], c=df['tms'], cmap='viridis')
plt.xlabel('Dividend to Price Ratio (dp)')
plt.ylabel('Dividend to Yields (d/y)')
plt.title('Dividend to Price Ratio vs. Dividend to Yields ')
_ = plt.colorbar(label='Market Risk Premium (tms)')

# %% [markdown]
# ##  Spike Analysis 

# %%
# Assuming 'R' column represents returns and the index is a datetime index.
def find_spike_periods(df, return_column='R', threshold=2):
    """
    Finds periods of spikes in returns exceeding a given threshold.

    Args:
        DF_FINAL: DataFrame with a datetime index and a return column.
        return_column: The name of the column containing returns.
        threshold: The standard deviation threshold to identify a spike.

    Returns:
        A list of tuples, where each tuple represents a spike period
        (start_date, end_date).
    """

    # Calculate rolling standard deviation to identify volatility
    rolling_std = DF_FINAL[return_column].rolling(window=12).std() # Adjust window size as needed

    # Identify spikes based on threshold
    spikes = DF_FINAL[return_column][rolling_std > threshold * rolling_std.mean()]

    # Group consecutive spikes into periods
    spike_periods = []
    start_date = None
    for date in spikes.index:
        if start_date is None:
            start_date = date
        elif date != spikes.index[spikes.index.get_loc(date) - 1] + pd.DateOffset(months=1): # Adjust for your data freq
            spike_periods.append((start_date, spikes.index[spikes.index.get_loc(date) - 1]))
            start_date = date
    if start_date is not None:
        spike_periods.append((start_date, spikes.index[-1]))

    return spike_periods

# Example usage:
spike_periods = find_spike_periods(DF_FINAL)
print(spike_periods)

# For visualization
plt.figure(figsize=(12, 6))
plt.plot(DF_FINAL['R'], label='Returns')
plt.plot(DF_FINAL['R'].rolling(window=6).std(), label='Rolling Std Dev')

for start, end in spike_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Spike Period' if start==spike_periods[0][0] else '') # Plot each spike as a shaded area
plt.legend()
plt.title('Return Spikes')
plt.xlabel('Date')
plt.ylabel('Return')
plt.show()

# %%

lags_mr = list(range(1, 4,1)) + list(range(6, 13, 6))
# Apply lags for 'mr'
for lag in lags_mr:
    x_train[f'mr_lag{lag}'] = x_train['mr'].shift(lag)
    x_test[f'mr_lag{lag}'] = x_test['mr'].shift(lag)

# Handle missing values
x_train = x_train.fillna(method='bfill')
x_test = x_test.fillna(method='bfill')
x_train.dropna(inplace=True)

x_test.dropna(inplace=True)

# %% [markdown]
# #### Analyse

# %% [markdown]
# ### Dropping columns

# %% [markdown]
# #### Drop Book to market ratio

# %%
x_train = x_train.drop('b/m', axis=1) 
x_test = x_test.drop('b/m', axis=1)

# %% [markdown]
# #### Drop long term yield 
# Dropping LTY and TBL because the relationship is covered in term spread 

# %%
x_train = x_train.drop('lty', axis=1) 
x_test = x_test.drop('lty', axis=1)

# %%
x_test.describe()

# %% [markdown]
# ## Data Transformation
# The evaluation of the data revealed a high standard deviation in certain features. To mitigate the potential impact of this variability and ensure features contribute equally to model training, data standardization was applied. This process transforms the data to have zero mean and unit variance, effectively balancing the dataset.

# %% [markdown]
# ### Data Scaling

# %%
scaler_x = StandardScaler()
columns = x_train.columns

x_train[columns] = scaler_x.fit_transform(x_train[columns])  # Fit on x_train, transform x_train
x_test[columns] = scaler_x.transform(x_test[columns])        # Transform x_test using the same scaler

# %% [markdown]
# ### Analyse Transformation

# %%
x_test.describe()

# %% [markdown]
# # Models

# %% [markdown]
# ## Linear Regression

# %% [markdown]
# ### OLS

# %% [markdown]
# #### Training

# %%
ols = LinearRegression()
ols = ols.fit(x_train, y_train)
y_insample_pred_ols = ols.predict(x_train)
y_outsample_pred_ols = ols.predict(x_test)

# %% [markdown]
# #### Analysing OLS Coeeficient 

# %%
# Assuming 'ols' is the fitted Linear Regression model from your code
# Access coefficients
coefficients = ols.coef_

# Access feature names
feature_names = x_train.columns

# Create a DataFrame for coefficients and their importance
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients[0]})

# Sort by absolute value of coefficients (importance)
coefficients_df['Abs_Coefficient'] = np.abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)
coefficients_df = coefficients_df.drop(columns=['Abs_Coefficient'])

# Display the coefficients and their importance
coefficients_df

# %% [markdown]
# ### Ridge

# %% [markdown]
# #### Time serires Cross validation

# %%
tscv = TimeSeriesSplit(n_splits=5, test_size=12 )

# %% [markdown]
# ####  Alpha Cross Validation

# %%
alphas = np.logspace(-4, 4, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring='neg_mean_squared_error')
ridge_cv.fit(x_train, y_train)

# %% [markdown]
# #### Training

# %%
ridge = Ridge(alpha=ridge_cv.alpha_,max_iter=1000,fit_intercept=False)
ridge.fit(x_train, y_train)
y_insample_pred_ridge = ridge.predict(x_train)
y_outsample_pred_ridge = ridge.predict(x_test)

# %% [markdown]
# ### Lasso

# %% [markdown]
# #### Time serires Cross validation

# %%
tscv = TimeSeriesSplit(n_splits=5, test_size=12)

# %% [markdown]
# #### Alpha Cross Validation

# %%
alphas = np.logspace(-4, 4, 100) 
lasso_cv = LassoCV(alphas=alphas, cv=tscv)
lasso_cv.fit(x_train, y_train)

# %% [markdown]
# #### Analysing the coefficent

# %%
# @ Analysing the coefficent
coefficients = lasso_cv.coef_

feature_names = x_train.columns
dropped_features = feature_names[np.where(coefficients ==0)]
print(dropped_features)
import numpy as np

importance = np.abs(coefficients)
sorted_indices = np.argsort(importance)[::-1]  # Indices sorted by importance

# If you have feature names (e.g., from a pandas DataFrame):
for i in sorted_indices:
    print(f"{feature_names[i]}: {importance[i]}")

# %% [markdown]
# #### Training

# %%
lasso = Lasso(alpha=lasso_cv.alpha_,max_iter=1000,fit_intercept=False)
lasso.fit(x_train, y_train)
y_insample_pred_lasso = lasso.predict(x_train)
y_outsample_pred_lasso = lasso.predict(x_test)

# %% [markdown]
# ## Random Forest

# %%

# Implement Random Forest with optimized hyperparameter tuning
rf = RandomForestRegressor(random_state=42, n_jobs=- 4)  # Use all CPU cores

rf_params = {
    "n_estimators": [50, 100, 200, 300],  # Fewer estimators for faster training
    "max_depth": [None, 10, 20, 30],     # Balanced depth options
    "min_samples_split": [2, 5, 10, 15], # Tuning sample split criteria
}

# Use fewer CV folds if speed is critical
rf_random = RandomizedSearchCV(
    rf, rf_params, cv=TimeSeriesSplit(n_splits=4, test_size=12), scoring="neg_mean_squared_error"
)


# Ensure y_train is a 1D array
y_train_1d = y_train.to_numpy().ravel()

# Fit the model
rf_random.fit(x_train, y_train_1d)

print("Best Random Forest Params:", rf_random.best_params_)

# Predict and evaluate Random Forest
y_train_pred_rf = rf_random.best_estimator_.predict(x_train)
y_test_pred_rf = rf_random.best_estimator_.predict(x_test)

# %% [markdown]
# # Performance Test

# %% [markdown]
# ## Performance Metrics

# %% [markdown]
# ### Define a function to calculate the performance metrics

# %%
import matplotlib.pyplot as plt
import pandas as pd

def timing_strategy_evaluation_with_drawdown(trained_model, X_test, actual_returns, risk_free_rate=0.02 / 12, threshold=0, initial_value=100):
    """
    Evaluate a timing strategy based on a trained model's predictions.

    Parameters:
        trained_model: Trained machine learning model with a `predict` method.
        X_test: DataFrame or array of predictors for testing (features for prediction).
        actual_returns: Series or array of actual returns for the evaluation period.
        risk_free_rate: Monthly risk-free rate, default is 0.02 annualized.
        threshold: Threshold for deciding risk-on or risk-off, default is 0.
        initial_value: Initial portfolio value, default is 100.

    Returns:
        portfolio_values: Series of portfolio values over time.
        cumulative_return: Final cumulative return of the portfolio.
        sharpe_ratio: Sharpe ratio of the portfolio strategy.
        max_drawdown: Maximum drawdown of the portfolio.
    """
    # Predict returns using the trained model
    predicted_returns = trained_model.predict(X_test)
    
    # Ensure actual_returns is a NumPy array for consistency
    if isinstance(actual_returns, pd.Series) or isinstance(actual_returns, pd.DataFrame):
        actual_returns = actual_returns.values.flatten()
    elif not isinstance(actual_returns, (list, tuple)):
        raise TypeError("actual_returns must be a Series, DataFrame, list, or tuple.")

    # Initialize portfolio for timing strategy
    portfolio_values_timing = [initial_value]

    # Timing strategy
    for i in range(len(predicted_returns)):
        if predicted_returns[i] > threshold:  # Risk-On
            portfolio_values_timing.append(portfolio_values_timing[-1] * (1 + actual_returns[i]))
        else:  # Risk-Off
            portfolio_values_timing.append(portfolio_values_timing[-1] * (1 + risk_free_rate))

    # Convert portfolio values to pandas Series for analysis
    portfolio_values_timing = pd.Series(portfolio_values_timing)

    # Calculate performance metrics for timing strategy
    cumulative_return_timing = portfolio_values_timing.iloc[-1] / portfolio_values_timing.iloc[0] - 1
    sharpe_ratio_timing = (portfolio_values_timing.pct_change().mean() - risk_free_rate) / portfolio_values_timing.pct_change().std()

    # Calculate maximum drawdown for timing strategy
    rolling_max_timing = portfolio_values_timing.cummax()
    drawdown_timing = (portfolio_values_timing - rolling_max_timing) / rolling_max_timing
    max_drawdown_timing = drawdown_timing.min()

    # Buy-and-hold strategy
    portfolio_values_bh = [initial_value]
    for ret in actual_returns:
        portfolio_values_bh.append(portfolio_values_bh[-1] * (1 + ret))
    portfolio_values_bh = pd.Series(portfolio_values_bh)

    # Plot portfolio evolution
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values_timing, label="Timing Strategy", marker='o', linestyle='-')
    plt.plot(portfolio_values_bh, label="Buy-and-Hold Strategy", marker='x', linestyle='--')
    plt.title("Portfolio Evolution: Timing Strategy vs Buy-and-Hold")
    plt.xlabel("Time (Months)")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()

    return portfolio_values_timing, cumulative_return_timing, sharpe_ratio_timing, max_drawdown_timing



# %% [markdown]
# ### Linear Regression 

# %% [markdown]
# #### In-sample Performance Comparism

# %%
mse_insample_ols = mean_squared_error(y_train, y_insample_pred_ols)
r2_insample_ols = r2_score(y_train, y_insample_pred_ols)
mse_insample_ridge = mean_squared_error(y_train, y_insample_pred_ridge)
r2_insample_ridge = r2_score(y_train, y_insample_pred_ridge)
mse_insample_lasso = mean_squared_error(y_train, y_insample_pred_lasso)
r2_insample_lasso = r2_score(y_train, y_insample_pred_lasso)
print('Model Insample Performance Comparison:')
print(f'OLS MSE: {mse_insample_ols:.4f}, R-squared: {r2_insample_ols:.4f}')
print(f'Ridge MSE: {mse_insample_ridge:.4f}, R-squared: {r2_insample_ridge:.4f}')
print(f'Lasso MSE: {mse_insample_lasso:.4f}, R-squared: {r2_insample_lasso:.4f}')


# %% [markdown]
# #### out-sample Performance Comparism

# %%
mse_outsample_ols = mean_squared_error(y_test, y_outsample_pred_ols)
r2_outsample_ols = r2_score(y_test, y_outsample_pred_ols)
mse_outsample_ridge = mean_squared_error(y_test, y_outsample_pred_ridge)
r2_outsample_ridge = r2_score(y_test, y_outsample_pred_ridge)
mse_outsample_lasso = mean_squared_error(y_test, y_outsample_pred_lasso)
r2_outsample_lasso = r2_score(y_test, y_outsample_pred_lasso)
print('Model Outsample Performance Comparison:')
print(f'OLS MSE: {mse_outsample_ols:.4f}, R-squared: {r2_outsample_ols:.4f}')
print(f'Ridge MSE: {mse_outsample_ridge:.4f}, R-squared: {r2_outsample_ridge:.4f}')
print(f'Lasso MSE: {mse_outsample_lasso:.4f}, R-squared: {r2_outsample_lasso:.4f}')

# %% [markdown]
# ### Timing Strategy for OLS
# 

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(ols, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ### Timing Strategy for Ridge

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(ridge, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ### Timing Strategy for Lasso
# 

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(lasso, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ### Random Forest

# %% [markdown]
# #### insample Performance comparism 

# %%
mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
r2_train_rf = r2_score(y_train, y_train_pred_rf)
print(f"Random Forest - Training MSE: {mse_train_rf:.4f}, Training R²: {r2_train_rf:.4f}")

# %% [markdown]
# #### out-sample Performance comparism

# %%
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print(f"Random Forest - Testing MSE: {mse_test_rf:.4f}, Testing R²: {r2_test_rf:.4f}")

# %% [markdown]
# #### Timing Strategy for Random Forest
# 

# %%
portfolio_values, cumulative_return, sharpe_ratio, max_drawdown = timing_strategy_evaluation_with_drawdown(rf_random, x_test, y_test)

# Display results
print("Cumulative Return:", round(cumulative_return * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown * 100, 2), "%")

# %% [markdown]
# ## Visuals

# %% [markdown]
# ### Plot for Linear Regression   
# 

# %%
# In-sample Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_train.index, y_train, label="Actual (Train)", color="blue")
plt.plot(y_train.index, y_insample_pred_ols, label="Predicted (Train)", color="orange")
plt.title("In-sample Predictions")
plt.xlabel("Date")
plt.ylabel("R")
plt.legend()
plt.show()

# Out-of-sample Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual (Test)", color="blue")
plt.plot(y_test.index, y_outsample_pred_ols, label="Predicted (Test)", color="green")
plt.title("Out-of-sample Predictions")
plt.xlabel("Date")
plt.ylabel("R")
plt.legend()
plt.show()


# %% [markdown]
# ### Plot for Random Forest
# 

# %%
plt.figure(figsize=(12, 6))

# In-sample plot
plt.subplot(1, 2, 1)
plt.plot(y_train.index, y_train, label='Actual', color='blue')
plt.plot(y_train.index, y_train_pred_rf, label='Predicted', color='red')
plt.title('Random Forest - In-sample Predictions')
plt.xlabel('Date')
plt.ylabel('R')
plt.legend()

# Out-of-sample plot
plt.subplot(1, 2, 2)
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_test_pred_rf, label='Predicted', color='green')
plt.title('Random Forest - Out-of-sample Predictions')
plt.xlabel('Date')
plt.ylabel('R')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# # Financial Analysis

# %% [markdown]
# ## Analysing data of Know Historical events

# %%


# %% [markdown]
# ## Market Valuation Signal

# %% [markdown]
# ### Dividend to Price Ratio vs. Book to Market Ratio
# The Graph shows strong correlation between the book to market ratio and Divends to price ratio

# %%
plt.figure(figsize=(8, 6))
plt.scatter(DF_FINAL['dp'], DF_FINAL['b/m'], c=DF_FINAL['tms'], cmap='viridis')
plt.xlabel('Dividend to Price Ratio (dp)')
plt.ylabel('Book to Market Ratio (b/m)')
plt.title('Dividend to Price Ratio vs. Book to Market Ratio')
_ = plt.colorbar(label='Market Risk Premium (tms)')

# %% [markdown]
# ### Dividend price vs Dividend Yield
# Dividend price vs Dividend Yield

# %%
plt.figure(figsize=(8, 6))
plt.scatter(DF_FINAL['dp'], DF_FINAL['dy'], c=df['tms'], cmap='viridis')
plt.xlabel('Dividend to Price Ratio (dp)')
plt.ylabel('Dividend to Yields (d/y)')
plt.title('Dividend to Price Ratio vs. Dividend to Yields ')
_ = plt.colorbar(label='Market Risk Premium (tms)')

# %% [markdown]
# ##  Spike Analysis 

# %%
# Assuming 'R' column represents returns and the index is a datetime index.
def find_spike_periods(df, return_column='R', threshold=2):
    """
    Finds periods of spikes in returns exceeding a given threshold.

    Args:
        DF_FINAL: DataFrame with a datetime index and a return column.
        return_column: The name of the column containing returns.
        threshold: The standard deviation threshold to identify a spike.

    Returns:
        A list of tuples, where each tuple represents a spike period
        (start_date, end_date).
    """

    # Calculate rolling standard deviation to identify volatility
    rolling_std = DF_FINAL[return_column].rolling(window=12).std() # Adjust window size as needed

    # Identify spikes based on threshold
    spikes = DF_FINAL[return_column][rolling_std > threshold * rolling_std.mean()]

    # Group consecutive spikes into periods
    spike_periods = []
    start_date = None
    for date in spikes.index:
        if start_date is None:
            start_date = date
        elif date != spikes.index[spikes.index.get_loc(date) - 1] + pd.DateOffset(months=1): # Adjust for your data freq
            spike_periods.append((start_date, spikes.index[spikes.index.get_loc(date) - 1]))
            start_date = date
    if start_date is not None:
        spike_periods.append((start_date, spikes.index[-1]))

    return spike_periods

# Example usage:
spike_periods = find_spike_periods(DF_FINAL)
print(spike_periods)

# For visualization
plt.figure(figsize=(12, 6))
plt.plot(DF_FINAL['R'], label='Returns')
plt.plot(DF_FINAL['R'].rolling(window=6).std(), label='Rolling Std Dev')

for start, end in spike_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Spike Period' if start==spike_periods[0][0] else '') # Plot each spike as a shaded area
plt.legend()
plt.title('Return Spikes')
plt.xlabel('Date')
plt.ylabel('Return')
plt.show()



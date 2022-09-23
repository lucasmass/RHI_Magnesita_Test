'''
Introduction
One of the preprocessing steps in machine learning is feature encoding. It is the process of turning categorical data in a dataset into numerical data. It is important that we perform feature encoding because most machine learning algorithms only handle numerical data and not data in text form.

We will learn the difference between nominal variables and ordinal variables. In addition, we will explore how OneHotEncoder and OrdinalEncoder can be used to transform these variables as part of a machine learning pipeline.

We will use this pipeline to predict the mean test score of different students. This is a regression problem in machine learning.
'''
# Import libraries
# Data wrangling
import pandas as pd
import numpy  as np

# Data visualisation
import seaborn           as sns
import matplotlib.pyplot as plt

# Machine learning
from sklearn.preprocessing   import OneHotEncoder
from sklearn.compose         import make_column_transformer
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Import and read data
data1 = pd.read_csv('/Users/lucasmassaroppe/Documents/rhi_magnesita/md_raw_dataset.csv',    sep=';')
data2 = pd.read_csv('/Users/lucasmassaroppe/Documents/rhi_magnesita/md_target_dataset.csv', sep=';') 

# Missing values, merge dataframes and data types
# making the 'Unamed: 0' column (from 'md_raw_dataset.csv' dataset) as one of the the keys to merge the dataframes
# the second key to merge both datasets is the column 'groups'
data1['index'] = data1['Unnamed: 0'] 

data = pd.merge(data1, data2, on=['index', 'groups'])

# droping unwanted columns
data = data.drop(columns=['crystal_type', 'crystal_supergroup', 'Cycle', 'Unnamed: 0', 'super_hero_group', 'when', 'Unnamed: 7', 
                          'Unnamed: 17', 'etherium_before_start', 'expected_start', 
                          'start_process', 'start_subprocess1', 'start_critical_subprocess1', 
                          'predicted_process_end', 'process_end', 'subprocess1_end', 
                          'reported_on_tower', 'opened', 'raw_kryptonite', 'index'])

# handling missing values by interpolation
data = data.interpolate(method='cubicspline')

'''
Exploratory data analysis (EDA)

Exploratory data analysis is the process of analysing and visualising the variables in a dataset.

Predictor variables
The predictor variables in the dataset are:

- super_hero_group;
- tracking;
- place;
- tracking_times;
- human_behavior_report;
- human_measure;
- crystal_weight;
- expected_factor_x;
- previous_factor_x;
- first_factor_x;
- expected_final_factor_x; 
- final_factor_x;
- previous_adamantium; 
- chemical_x; 
- argon; 
- pure_seastone;
- groups;

In this section, we will explore how these different features influence the outcome of the 'target' test score.
'''

# Correlation Matrix

data.corr()

# As we can see from the correlation structure, the predictors have a dependency on each other and on the target column, making it difficult to reduce the dimensionality of this dataset.

'''
Build machine learning pipeline

A pipeline chains together multiple steps in the machine learning process where the output of each step is used as input to the next step. It is typically used to chain data preprocessing procedures together with modelling into one cohesive workflow.

Here, we will build two pipelines that share the same column transformer that we have created above but with a different machine learning model, one using linear regression and the other using gradient boosting.

We will then compare the accuracy of the prediction results using mean absolute error (MAE) as well as root mean squared error (RMSE). The model with a lower prediction error is deemed more accurate than the other.
'''

# Train test split 

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)

# Instantiate pipeline with linear regression

lm = LinearRegression()
lm_pipeline = make_pipeline(lm)

# Instantiate pipeline with gradient boosting

gbm = GradientBoostingRegressor()
gbm_pipeline = make_pipeline(gbm)

# Fit pipeline to training set and make predictions on test set 

lm_pipeline.fit(X_train, Y_train)
lm_predictions = lm_pipeline.predict(X_test)

# As we can see, the gradient boosting regression method performs better than the linear regression method.

plt.scatter(gbm_predictions, Y_test)
plt.xlabel('True Values ')
plt.ylabel('Predictions ')
plt.axis('equal')
plt.axis('square')
plt.show()

gbm_pipeline.fit(X_train, Y_train)
gbm_predictions = gbm_pipeline.predict(X_test)

# Calculate mean square error and root mean squared error 

lm_mae = mean_absolute_error(lm_predictions, Y_test)
lm_rmse = np.sqrt(mean_squared_error(lm_predictions, Y_test))
print("LM MAE: {:.2f}".format(round(lm_mae, 2)))
print("LM RMSE: {:.2f}".format(round(lm_rmse, 2)))

gbm_mae = mean_absolute_error(gbm_predictions, Y_test)
gbm_rmse = np.sqrt(mean_squared_error(gbm_predictions, Y_test))
print("GBM MAE: {:.2f}".format(round(gbm_mae, 2)))
print("GBM RMSE: {:.2f}".format(round(gbm_rmse, 2)))
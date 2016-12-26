#!/usr/bin/env python
"""
- House prices example from: https://github.com/ClimbsRocks/auto_ml
"""

import dill
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from auto_ml import Predictor

# Load data
boston = load_boston()
df_boston = pd.DataFrame(boston.data)
df_boston.columns = boston.feature_names
df_boston['MEDV'] = boston['target']
df_boston_train, df_boston_test = train_test_split(df_boston, test_size=0.2, random_state=42)

# Tell auto_ml which column is 'output'
# Also note columns that aren't purely numerical
# Examples include ['nlp', 'date', 'categorical', 'ignore']
column_descriptions = {
  'MEDV': 'output'
  , 'CHAS': 'categorical'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df_boston_train)

# Score the model on test data
test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

# auto_ml is specifically tuned for running in production
# It can get predictions on an individual row (passed in as a dictionary)
# A single prediction like this takes ~1 millisecond
# Here we will demonstrate saving the trained model, and loading it again
file_name = ml_predictor.save()

# dill is a drop-in replacement for pickle that handles functions better
with open (file_name, 'rb') as read_file:
    trained_model = dill.load(read_file)

# .predict and .predict_proba take in either:
# A pandas DataFrame
# A list of dictionaries
# A single dictionary (optimized for speed in production evironments)
predictions = trained_model.predict(df_boston_test)
print(predictions)

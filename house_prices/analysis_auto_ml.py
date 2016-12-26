#!/usr/bin/env python
"""


analysis_auto_ml.py
=====================
Built for kaggle-houseprices:
- https://github.com/omahapython/kaggle-houseprices


auto_ml
---------
| Src: https://github.com/ClimbsRocks/auto_ml
| Docs: https://auto-ml.readthedocs.io/en/latest/
| Docs: https://auto-ml.readthedocs.io/en/latest/api_docs_for_geeks.html

- House prices example from: https://github.com/ClimbsRocks/auto_ml

Installation
--------------

.. code:: bash

    conda install scikit-learn pandas dill
    pip install xgboost auto_ml

Usage
------

.. code:: bash

    python ./analysis_auto_ml.py
    wc -l ~/data/submission_auto_ml.csv

"""

import dill
import pandas as pd
from sklearn.model_selection import train_test_split

from auto_ml import Predictor

# Load data

from house_prices.data import HousePricesSuperBunch

sb = HousePricesSuperBunch.load(
    cfg=dict(
        do_get_dummies=False,
        do_categoricals=False,
        do_autoclean=False)
)

houseprices = sb.get_train_bunch(sparse=True)

# houseprices = load_houseprices()
# df_houseprices = pd.DataFrame(houseprices.data)
# df_houseprices.columns = houseprices.feature_names
# df_houseprices['MEDV'] = houseprices['target']

df_houseprices = sb.train_df
df_houseprices_train, df_houseprices_test = train_test_split(df_houseprices, test_size=0.2, random_state=42)
df_houseprices_train = sb.train_df
#df_houseprices_test = sb.test_df

# Tell auto_ml which column is 'output'
# Also note columns that aren't purely numerical
# Examples include ['nlp', 'date', 'categorical', 'ignore']
column_descriptions = {
    'SalePrice': 'output'
}
for column in sb.train_schema.categorical_columns:
    column_descriptions[column] = 'categorical'

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df_houseprices_train,
                   take_log_of_y=True,
                   #compute_power=10,
                   )

# Score the model on test data
test_score = ml_predictor.score(df_houseprices_test, df_houseprices_test['SalePrice'])

# auto_ml is specifically tuned for running in production
# It can get predictions on an individual row (passed in as a dictionary)
# A single prediction like this takes ~1 millisecond
# Here we will demonstrate saving the trained model, and loading it again
file_name = ml_predictor.save()

# dill is a drop-in replacement for pickle that handles functions better
with open(file_name, 'rb') as read_file:
    trained_model = dill.load(read_file)

# .predict and .predict_proba take in either:
# A pandas DataFrame
# A list of dictionaries
# A single dictionary (optimized for speed in production evironments)
# predictions = trained_model.predict(df_houseprices_test)
# print(predictions)


#print(sb.test_df)
predictions = trained_model.predict(sb.test_df)

submission_df = pd.DataFrame(
    predictions,
    index=sb.test_df.index,
    columns=['SalePrice'])
print(submission_df)
submission_df.to_csv('data/submission_auto_ml.csv', index_label='Id')

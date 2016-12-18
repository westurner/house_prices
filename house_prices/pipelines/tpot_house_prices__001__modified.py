
"""
https://github.com/westurner/house_prices/commit/3f586b57ac081a9bef27f43906b3df83a27f0b45
  - do_autoclean=True
  - do_get_dummies=False
"""
# import numpy as np
from collections import OrderedDict as odict

from datacleaner import autoclean_cv
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer


class PredictionPipeline(object):
    def __init__(self,
                 train_csv='train.csv',
                 test_csv='test.csv',
                 predict_csv='predict.csv',
                 classcolname='class'):
        self.cfg = odict()
        self.cfg['train_csv'] = train_csv
        self.cfg['test_csv'] = test_csv
        self.cfg['predict_csv'] = predict_csv
        self.cfg['classcolname'] = classcolname
        self.data = odict()

    def fit(self, **kwargs):
        train_csv = kwargs.get('train_csv', self.cfg['train_csv'])
        test_csv = kwargs.get('test_csv', self.cfg['test_csv'])
        classcolname = kwargs.get('classcolname', self.cfg['classcolname'])
        # TODO: log
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        test_df[classcolname] = 0
        train_df_cleaned, test_df_cleaned = autoclean_cv(train_df, test_df,
                                                        ignore_update_check=True)
        del test_df[classcolname] # TODO: comment_these_xyz !?

        # features = np.delete(train_df_cleaned.view(np.float64).reshape(train_df_cleaned.size, -1), train_df_cleaned.dtype.names.index('class'), axis=1)

        train_class = train_df_cleaned[classcolname]

        del train_df_cleaned[classcolname]  # TODO: comment_these_xyz !?
        train_features = train_df_cleaned.as_matrix()

        training_features, testing_features, training_classes, testing_classes = \
            train_test_split(train_features, train_class, random_state=42)

        self.exported_pipeline = make_pipeline(
            make_union(
                make_union(VotingClassifier([('branch',
                    ElasticNet(alpha=1.0, l1_ratio=0.87)
                )]), FunctionTransformer(lambda X: X)),
                FunctionTransformer(lambda X: X)
            ),
            make_union(VotingClassifier([("est", GradientBoostingRegressor(learning_rate=0.02, max_features=0.02, n_estimators=500))]), FunctionTransformer(lambda X: X)),
            ExtraTreesRegressor(max_features=0.27, n_estimators=500)
        )

        self.exported_pipeline.fit(training_features, training_classes)

        self.data['train_df_cleaned'] = train_df_cleaned
        self.data['test_df_cleaned'] = test_df_cleaned
        self.data['train_features'] = train_features
        self.data['train_class'] = train_class

    def output_heuristics(self, predict_df):
        classcolname = self.cfg['classcolname']
        invalid_predict_df = predict_df[predict_df[classcolname] <= 0]
        if len(invalid_predict_df):
            print(invalid_predict_df)
            raise Exception()
        if len(self.data["test_df_cleaned"]) != len(predict_df):
            raise Exception()

    def predict_with_test(self):
        classcolname = self.cfg['classcolname']
        test_df_cleaned = self.data['test_df_cleaned']
        results = self.exported_pipeline.predict(test_df_cleaned.as_matrix())
        predict_df = pd.DataFrame(results, columns=[classcolname])
        predict_df.to_csv('../data/submission.csv',
                          index_label="Id")

        self.output_heuristics(predict_df)

    def predict_with_train(self):
        train_features = self.data['train_features']
        train_class = self.data['train_class']

        results = self.exported_pipeline.predict(train_features)
        train_results_df = pd.DataFrame(results, columns=['predicted'])
        train_results_df['actual'] = train_class
        train_results_df['diff'] = (train_results_df['predicted'] - train_results_df['actual'])
        # print(train_results_df['diff'])
        # print(train_results_df[train_results_df['diff'] != 0])
        class_sum = train_results_df['actual'].sum()
        print('class_sum:', class_sum)
        abs_error = train_results_df['diff'].abs().sum()
        print('abs error:', abs_error)
        percent_error = 100.0 * (1.0 - ((class_sum - abs_error) / class_sum))
        print('% error:  ', percent_error, '%')
        print('error**2: ', (train_results_df['diff']**2).sum())

    def run(self):
        self.fit()
        self.predict_with_test()
        self.predict_with_train()


def main():
    analysis = PredictionPipeline(
        train_csv='../data/train.csv',
        test_csv='../data/test.csv',
        predict_csv='../data/predict.csv',
        classcolname='SalePrice')
    analysis.run()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

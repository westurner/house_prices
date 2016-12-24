#!/usr/bin/env python
"""
| Src: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/base.py
| License: BSD 3-Clause https://github.com/scikit-learn/scikit-learn/blob/master/COPYING


"""

import re

from collections import OrderedDict as odict
from os.path import join, dirname

import datacleaner
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch


class DataMeta(object):  # TODO: DataMeta(OrderedDict)? [key] // .attr
    def __init__(self,
                 schema,
                 categorical_columns=None,
                 description=None):
        self.schema = schema
        self.categorical_columns = categorical_columns
        self.description = description


class DataSet(object):
    def __init__(self, data, meta=None):
        self.data = data
        if meta is None:
            meta = odict()
            meta = DataMeta()
        self.meta = meta

    # def to_matrix(self):
    #     return self.data

    # def __getattribute__(self, attr):
    #     if hasattr(DataSet, attr):
    #         return DataSet.__dict__(attr)
    #     return getattr(self.data, attr)

    # def __setattribute__(self, attr, val):
    #     if hasattr(self.data):
    #         return setattr(self.data, attr, val)
    #     return setattr(self, attr, val) # TODO


# class DataFrameDataSet(DataSet):
#     def to_matrix(self):
#         return self.data.to_matrix()


# class CSVDataFrameDataSet(DataFrameDataSet):
#     pass


class SuperBunch(object):
    """
        - TODO: specify multiple training datasets?
    """
    def __init__(self,
                 #train_datasets,
                 #test_datasets,
                 train_df,
                 train_schema=None,
                 train_features=None,
                 train_description=None,
                 test_df=None,
                 test_schema=None,
                 test_features=None,
                 test_description=None,
                 cfg=None):
        self.cfg = odict()
        if cfg is not None:
            self.cfg.update(cfg)

        self.train_df = train_df
        self.train_schema = train_schema

        self.test_df = test_df
        self.test_schema = train_schema  # TODO

    @property
    def train_features(self):
        self.train_df.columns.tolist()

    @property
    def test_features(self):
        self.test_df.columns.tolist()

    def get_bunch(self, df, description=None, predict_colname=None,
                  target=None,
                  sparse=False):
        if predict_colname is None:
            predict_colname = self.cfg['predict_colname']
        columns = [x for x in df.columns if x != predict_colname]
        if not sparse:
            data = df[columns].as_matrix()
            if target is None:
                target = df[predict_colname].as_matrix()
        else:
            data = df[columns].to_sparse()
            if target is None:
                target = df[predict_colname].to_sparse()
        return Bunch(
            data=data,
            target=target,
            feature_names=columns,
            DESCR=description)

    def get_train_bunch(self, predict_colname=None, **kwargs):
        return self.get_bunch(self.train_df,
                              description=self.train_schema.description,
                              predict_colname=predict_colname,
                              **kwargs)

    def get_test_bunch(self, predict_colname=None, **kwargs):
        return self.get_bunch(self.test_df,
                              description=self.test_schema.description,
                              target=kwargs.get('target', np.array([])),
                              predict_colname=predict_colname,
                              **kwargs)


class HousePricesSuperBunch(SuperBunch):

    data_path = join(dirname(__file__), 'data')
    default_cfg = odict((
        ('do_categoricals', True),
        ('do_get_dummies', False),
        ('do_autoclean', 'drop'),
        ('predict_colname', 'SalePrice'),
    ))

    @classmethod
    def load(cls, cfg=None):
        _cfg = cls.default_cfg.copy()
        if cfg is not None:
            _cfg.update(cfg)
        sb = cls.from_csv(
            test_csv=join(cls.data_path, 'test.csv'),
            train_csv=join(cls.data_path, 'train.csv'),
            description_txt=join(cls.data_path, 'data_description.txt'),
            index_col='Id',
            cfg=_cfg
        )
        sb.fit_transform()
        return sb

    @classmethod
    def from_csv(cls,
                 train_csv=None,
                 test_csv=None,
                 description_txt=None,
                 index_col=0,
                 cfg=None):
        train_schema = cls.load_schema(description_txt)
        train_schema.train_csv = train_csv
        train_schema.test_csv = test_csv
        train_df = pd.read_csv(train_csv, index_col=index_col)
        test_df = pd.read_csv(test_csv, index_col=index_col)
        return cls(
            train_df,
            train_schema=train_schema,
            test_df=test_df,
            test_schema=train_schema, # TODO
            cfg=cfg)

    @staticmethod
    def parse_description(fileobj):
        feat_rgx = re.compile('^(?P<feature>\w+): (?P<description>.*)')
        levl_rgx = re.compile('^\s+(?P<name>\w+)\t(?P<label>.*)')

        featurelevels = odict()
        levl_list = None
        prev_feat = None
        for line in fileobj:
            feat_mobj = feat_rgx.match(line)
            if feat_mobj:
                feat_mdict = feat_mobj.groupdict()
                if levl_list is not None and prev_feat is not None:
                    featurelevels[prev_feat] = levl_list
                prev_feat = feat_mdict['feature']
                levl_list = []
            else:
                levl_mobj = levl_rgx.match(line)
                if levl_mobj:
                    levl_list.append([x.strip() for x in levl_mobj.groups()])
        if levl_list is not None and prev_feat is not None:
            featurelevels[prev_feat] = levl_list
        return featurelevels

    @staticmethod
    def keys_with_values(mapping):
        for colkey in mapping:
            values = mapping[colkey]
            if len(values):
                yield colkey

    @classmethod
    def load_schema(cls, description_filename='data_description.txt'):
        fdescr_name = join(cls.data_path, description_filename)
        with open(fdescr_name) as f:
            description = f.read()
            f.seek(0)
            schema = cls.parse_description(f) # column_categories
            categorical_columns = list(cls.keys_with_values(schema))
            return DataMeta(
                schema,
                categorical_columns,
                description)

    def fit_transform(self):
        # TODO: standardize, normalize, sklearn-pandas DataFrameMapper
        do_get_dummies = self.cfg.get('do_get_dummies')
        do_autoclean = self.cfg.get('do_autoclean')
        do_categoricals = self.cfg.get('do_categoricals')
        self.do_concatenate()
        if do_categoricals:
            self.do_categoricals()
        if do_get_dummies:
            self.do_get_dummies()
        if do_autoclean:
            self.do_autoclean()
        self.do_unconcatenate()
        self.write_transformed_data()

    def do_concatenate(self):
        # self.train_df['type'] = 'train'
        # self.test_df['type'] = 'test'
        self.train_schema.rows = len(self.train_df)
        self.train_df = pd.concat([self.train_df, self.test_df])
        self.test_df = None

    def do_unconcatenate(self):
        self.test_df = self.train_df[self.train_schema.rows:]
        self.train_df = self.train_df[:self.train_schema.rows]

    def do_categoricals(self):
        schema = self.train_schema
        dataframes = [df for df in [self.train_df, self.test_df]
                      if df is not None]
        for colname in schema.categorical_columns:
            categories = [x[0] for x in schema.schema[colname]]
            for df in dataframes:
                df[colname] = df[colname].astype('category',
                                        categories=categories,
                                        ordered=True) #TODO

    def do_get_dummies(self):
        schema = self.train_schema
        get_dummies_dict = {key: key for key in schema.categorical_columns}
        train_df = pd.get_dummies(self.train_df,
                            prefix=get_dummies_dict,
                            columns=get_dummies_dict)
        self.train_df = train_df
        if self.test_df is not None:
            test_df = pd.get_dummies(self.test_df,
                                     prefix=get_dummies_dict,
                                     columns=get_dummies_dict)
            self.test_df = test_df

    def do_autoclean(self):
        if self.test_df is not None:
            self.train_df, self.test_df = do_autoclean_cv(
                self.train_df, self.test_df,
                do_autoclean=self.cfg.get('do_autoclean'),
                predict_colname=self.cfg['predict_colname'])
        else:
            self.train_df = datacleaner.autoclean(
                self.train_df,
                ignore_update_check=True)

    def write_transformed_data(self):
        train_csv_transformed = (
            self.train_schema.train_csv + '.transformed.csv')
        self.train_df.to_csv(train_csv_transformed)
        if self.test_df is not None:
            test_csv_transformed = (
                self.train_schema.test_csv + '.transformed.csv')
            self.test_df.to_csv(test_csv_transformed)


def do_autoclean_cv(train_df, test_df,
                    do_autoclean='drop',
                    predict_colname=None):
    if do_autoclean and predict_colname is None:
        raise TypeError("predict_colname must be specified")
    if do_autoclean == 'drop':
        target_col = None
        if predict_colname in train_df:
            target_col = train_df[predict_colname]
            del train_df[predict_colname]
    elif do_autoclean == 'append_mean':
        test_df[predict_colname] = train_df[predict_colname].mean()
    elif do_autoclean == 'append_nan':
        test_df[predict_colname] = np.NaN

    try:
        train_df, test_df = datacleaner.autoclean_cv(
            train_df, test_df, ignore_update_check=True)
    except ValueError:
        print(train_df.columns.tolist())
        print(test_df.columns.tolist())
        print(set(train_df.columns).difference(set(test_df.columns)))
        raise

    if do_autoclean == 'drop':
        if target_col is not None:
            train_df[predict_colname] = target_col
    elif do_autoclean == 'append_mean':
        del test_df[predict_colname]
    elif do_autoclean == 'append_nan':
        del test_df[predict_colname]
    return train_df, test_df


def load_house_prices(return_X_y=False,
                      return_dataframes=False,
                      data_file='train.csv',
                      test_data_file=None,

                      do_categoricals=True,
                      do_autoclean='drop',
                      predict_colname='SalePrice',
                      do_get_dummies=False,

                      write_transformed_data=True):
    """Load and return the Kaggle Ames Iowa House Prices dataset.

    ==============     =======================
    Samples total                         1460
    Dimensionality                          81
    Features           real, positive, strings
    Targets                real 34900 - 755000
    ==============     =======================
    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    return_dataframes: boolean, default=False
        If true, returns ``(df)`` or ``(df, test_df)``
        instead of a Bunch object.
    data_file : str, default='train.csv'
        The data file to load
    test_data_file : str, default=None
        The test data file to load (e.g. test.csv)

    do_categoricals : bool, default=True
        If True, call df[col].astype('category', categories=[...], ordered=True)
        with each column
    do_autoclean : bool,str, default='drop'
        Whether to datacleaner.autoclean[_cv] the dataset(s)

        - 'drop': drop the predict_colname from train_df before autoclean
        - 'append_mean': test_df[predict_colname]=train_df[predict_colname].mean()
          before autoclean
    predict_colname : str, default='SalePrice'
        The column name of the column to be predicted
    do_get_dummies : bool, default=False
        Whether to run pd.do_get_dummies the dataset(s)
    write_transformed_data: bool, default=True
        If True, write transformed data in a file named with a
        'transformed.csv' suffix

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the regression targets,
        and 'DESCR', the full description of the dataset.
    (data, target) : tuple if ``return_X_y`` is True

    Examples
    --------
    >>> from house_prices.data import load_house_prices
    >>> house_prices = load_house_prices()
    >>> print(house_prices.data.shape)
    (1460, 81)
    """
    module_path = dirname(__file__)

    description_filepath = 'data_description.txt'
    fdescr_name = join(module_path, 'data', description_filepath)
    with open(fdescr_name) as f:
        descr_text = f.read()
        f.seek(0)
        column_categories = HousePricesSuperBunch.parse_description(f)

    data_file_name = join(module_path, 'data', data_file)
    df = pd.read_csv(data_file_name, index_col='Id')

    if test_data_file:
        test_data_file_name = join(module_path, 'data', test_data_file)
        test_df = pd.read_csv(test_data_file_name, index_col='Id')

    # TODO
    # if do_categoricals:
    #     HousePricesSuperBunch.do_categoricals(train_df=train_df, test_df=test_df)

    if do_get_dummies:
        def keys_with_values(column_categories):
            for colkey in column_categories:
                values = column_categories[colkey]
                if len(values):
                    yield colkey
        categorical_columns = list(keys_with_values(column_categories))
        get_dummies_dict = {key: key for key in categorical_columns}
        df = pd.get_dummies(df,
                            prefix=get_dummies_dict,
                            columns=get_dummies_dict)
        if test_data_file:
            test_df = pd.get_dummies(test_df,
                                     prefix=get_dummies_dict,
                                     columns=get_dummies_dict)

    if do_autoclean:
        if test_data_file:
            df, test_df = do_autoclean_cv(df, test_df,
                                          do_autoclean=do_autoclean,
                                          predict_colname=predict_colname)
        else:
            df = datacleaner.autoclean(df, ignore_update_check=True)

    if write_transformed_data:
        transformed_data_filename = (
            data_file_name + '.transformed.csv' if write_transformed_data is True
            else write_transformed_data)
        df.to_csv(transformed_data_filename)
        if test_data_file:
            clean_test_data_filename = test_data_file_name + '.transformed.csv'
            test_df.to_csv(clean_test_data_filename)

    feature_names = df.columns.tolist()
    target = df['SalePrice'].as_matrix()
    del df['SalePrice']
    if return_dataframes:
        if test_data_file is None:
            return df
        else:
            return df, test_df

    data = df.as_matrix()
    if test_data_file is None:
        if return_X_y:
            return data, target

        return Bunch(data=data,
                    target=target,
                    # last column is target value
                    feature_names=feature_names[:-1],
                    DESCR=descr_text)
    elif test_data_file:
        if return_X_y:
            return (data, target), (test_df.as_matrix(), None)
        return (Bunch(data=data,
                     target=target,
                     feature_names=feature_names[:-1],
                     DESCR=descr_text),
               Bunch(data=test_df.as_matrix(),
                     target=None,
                     feature_names=test_df.columns.tolist(),
                     DESCR=descr_text))

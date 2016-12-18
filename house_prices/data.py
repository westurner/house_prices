#!/usr/bin/env python
"""
| Src: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/base.py
| License: BSD 3-Clause https://github.com/scikit-learn/scikit-learn/blob/master/COPYING
"""

import re

from collections import OrderedDict as odict
from os.path import join, dirname

import datacleaner
import pandas as pd
from sklearn.datasets.base import Bunch


def parse_description(filename='data_description.txt'):
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'data', 'data_description.txt')

    rgx_var = re.compile('^(?P<feature>\w+): (?P<description>.*)')
    rgx_cat = re.compile('^\s+(?P<name>\w+)\t(?P<label>.*)')

    data_dict = odict()
    cur_category = None
    with open(fdescr_name) as f:
        for line in f:
            mobj = rgx_var.match(line)
            if mobj:
                feature_mdict = mobj.groupdict()
                if cur_category is not None:
                    data_dict[feature_mdict['feature']] = cur_category
                cur_category = []
            else:
                mobj = rgx_cat.match(line)
                if mobj:
                    mdict = mobj.groups()
                    cur_category.append(mdict)
        if cur_category is not None:
            data_dict[feature_mdict['feature']] = cur_category
    return data_dict


def load_house_prices(return_X_y=False, data_file='train.csv',
                      do_autoclean=True,
                      do_get_dummies=False,

                      write_clean_data=True):
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

    column_categories = parse_description(description_filepath)

    data_file_name = join(module_path, 'data', data_file)
    df = pd.read_csv(data_file_name)
    feature_names = df.columns.tolist()
    target = df['SalePrice'].as_matrix()
    del df['SalePrice']

    if do_get_dummies:
        def get_categorical_columns(column_categories):
            for colkey in column_categories:
                values = column_categories[colkey]
                if len(values):
                    yield colkey
        categorical_columns = list(get_categorical_columns(column_categories))
        get_dummies_dict = {key: key for key in categorical_columns}
        df = pd.get_dummies(df, prefix=get_dummies_dict, columns=get_dummies_dict)

    if do_autoclean:
        df = datacleaner.autoclean(df, ignore_update_check=True)

    if write_clean_data:
        clean_data_filename = (
            data_file_name + '.cleaned.csv' if write_clean_data is True
            else write_clean_data)
        df.to_csv(clean_data_filename)

    data = df.as_matrix()
    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text)

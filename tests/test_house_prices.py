#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_house_prices
----------------------------------

Tests for `house_prices` module.
"""

import pytest

from click.testing import CliRunner
from house_prices import cli


@pytest.mark.skip(reason="currently calls long-running analysis:main")
def test_command_line_interface():

    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'house_prices.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


# def test_parse_description():
#     from house_prices.data import parse_description
#     data = parse_description()
#     print(data)
#     for k,v in data.items():
#         print(k)
#         if isinstance(v, list):
#             for value in v:
#                 print(value)
#     raise Exception()


def test_load_house_prices():
    import numpy as np
    from sklearn.datasets.base import Bunch
    from house_prices.data import load_house_prices
    train = load_house_prices()
    assert isinstance(train, Bunch)
    assert isinstance(train.data, np.ndarray)
    assert isinstance(train.target, np.ndarray)
    assert isinstance(train.feature_names, list)


def test_load_house_prices_test_data__bunch():
    import numpy as np
    from sklearn.datasets.base import Bunch
    from house_prices.data import load_house_prices
    train, test = load_house_prices(
        test_data_file='test.csv',
        do_get_dummies=False)
    assert isinstance(train, Bunch)
    assert isinstance(train.data, np.ndarray)
    assert isinstance(train.target, np.ndarray)
    assert isinstance(train.feature_names, list)
    assert isinstance(test, Bunch)
    assert isinstance(test.data, np.ndarray)
    #assert isinstance(test.target, None)  # kaggle-specific
    assert isinstance(test.feature_names, list)
    assert train.data.shape == (1460, 79)
    assert test.data.shape == (1459, 79)


def test_load_house_prices_test_data__dataframes():
    from pandas import DataFrame
    from house_prices.data import load_house_prices
    train, test = load_house_prices(
        test_data_file='test.csv',
        return_dataframes=True)
    assert isinstance(train, DataFrame)
    assert isinstance(test, DataFrame)


def test_load_house_prices_class():
    import numpy as np
    from collections import OrderedDict as odict
    from sklearn.datasets.base import Bunch
    from house_prices.data import SuperBunch, HousePricesSuperBunch
    cfg = odict((
        ('do_categoricals', True),
        ('do_get_dummies', True),
        ('do_autoclean', 'drop'),
        ('predict_colname', 'SalePrice'),
        ('index_colname', 'Id'),
    ))

    data = HousePricesSuperBunch.load(cfg=cfg)
    assert isinstance(data, HousePricesSuperBunch)
    assert isinstance(data, SuperBunch)
    assert hasattr(data, 'cfg')
    assert hasattr(data.cfg, 'items')
    assert data.cfg == cfg
    assert data.cfg['do_get_dummies'] is True
    assert data.cfg['do_autoclean'] is 'drop'

    train_bunch = data.get_train_bunch()
    assert train_bunch.data.shape == (1460, 344)
    assert isinstance(train_bunch.target, np.ndarray)
    assert hasattr(data, 'train_features')
    assert hasattr(data, 'train_schema')
    assert isinstance(train_bunch, Bunch)

    test_bunch = data.get_test_bunch()
    assert test_bunch.data.shape == (1459, 344)
    assert isinstance(test_bunch, Bunch)
    assert hasattr(data, 'test_features')
    assert hasattr(data, 'test_schema')

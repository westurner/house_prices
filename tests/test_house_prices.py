#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_house_prices
----------------------------------

Tests for `house_prices` module.
"""

import pytest

from contextlib import contextmanager
from click.testing import CliRunner

from house_prices import house_prices
from house_prices import cli


@pytest.fixture
def response():
    """Sample pytest fixture.
    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument.
    """
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
def test_command_line_interface():
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'house_prices.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_parse_description():
    from house_prices.data import parse_description
    data = parse_description()
    print(data)
    for k,v in data.items():
        print(k)
        if isinstance(v, list):
            for value in v:
                print(value)
    raise Exception()


def test_load_house_prices():
    from house_prices.data import load_house_prices
    data = load_house_prices()
    print(data)
    raise Exception()

# -*- coding: utf-8 -*-

import click

from analysis import HousePricesAnalysis

@click.command()
def main(args=None):
    """Console script for house_prices"""
    click.echo("Replace this message by putting your code into "
               "house_prices.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")

    analysis = HousePricesAnalysis()
    analysis()
    print(analysis)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

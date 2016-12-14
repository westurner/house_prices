#!/usr/bin/env python
"""
| Src: https://rhiever.github.io/tpot/examples/Boston_Example/
| Src: https://github.com/rhiever/tpot/blob/master/docs/sources/examples/Boston_Example.md
| License: GPLv3 https://github.com/rhiever/tpot/blob/master/LICENSE
"""

import json

from collections import OrderedDict as odict
from os.path import join, dirname

from tpot import TPOTRegressor
from data import load_house_prices
from sklearn.model_selection import train_test_split


def do_analysis(**kwargs):

    data = odict()
    house_prices = load_house_prices()

    tts_kwargs = odict()
    tts_kwargs['train_size'] = kwargs.pop('train_size', 0.75)
    tts_kwargs['test_size'] = kwargs.pop('test_size', 0.25)
    data.update(tts_kwargs)
    X_train, X_test, y_train, y_test = train_test_split(
        house_prices.data, house_prices.target, **tts_kwargs)

    regressor_kwargs = odict()
    regressor_kwargs['generations'] = kwargs.pop('generations', 5)
    regressor_kwargs['population_size'] = kwargs.pop('population_size', 20)
    regressor_kwargs['verbosity'] = kwargs.pop('verbosity', 2)
    data.update(regressor_kwargs)
    tpot = TPOTRegressor(**regressor_kwargs)
    tpot.fit(X_train, y_train)
    data['score'] = tpot.score(X_test, y_test)
    print(('score', data['score']))

    export_dirpath = kwargs.pop('export_dirpath',
                            join(dirname(__file__), 'pipeline'))
    export_filename = kwargs.pop('export_filename', 'tpot_house_prices_.py')
    export_filepath = kwargs.pop('export_filepath',
                                 join(export_dirpath, export_filename))
    # data['export_filename'] = export_filename
    # data['export_dirpath'] = export_dirpath
    data['export_filepath'] = export_filepath
    tpot.export(export_filepath)

    data['export_filepath_datajson'] = export_filepath + '.json'
    with open(data['export_filepath_datajson'], 'w') as f:
        f.write(json.dumps(data))

    return data


def main():
    data = do_analysis()
    print(data)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

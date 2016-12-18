#!/usr/bin/env python
"""
| Src: https://rhiever.github.io/tpot/examples/Boston_Example/
| Src: https://github.com/rhiever/tpot/blob/master/docs/sources/examples/Boston_Example.md
| License: GPLv3 https://github.com/rhiever/tpot/blob/master/LICENSE
"""

import json
import logging
import os
import os.path

from collections import OrderedDict as odict
from os.path import join, dirname

from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

log = logging.getLogger(__file__)

class TPOTAnalysis(object):
    """
    A base class for tpot analyses.

    Subclasses should override FILENAME_PREFIX and dataloader()
    """

    FILENAME_PREFIX = 'tpot_pipeline'

    def __init__(self,
                 export_filename_prefix=None,
                 dataloader=None):
        self.data = odict()
        if export_filename_prefix is None:
            self.export_filename_prefix = self.FILENAME_PREFIX
        else:
            self.export_filename_prefix = export_filename_prefix

    def dataloader(self):
        raise NotImplementedError("This should be defined in a subclass."
                                  "It should return a sklean.base.Bunch")

    @property
    def filename_prefix(self):
        return self.export_filename_prefix

    def __call__(self, **kwargs):
        """
        Keyword Arguments are passed through to do_analysis().
        """
        if 'dataloader' not in kwargs:
            kwargs['dataloader'] = self.dataloader
        if 'export_filename_prefix' not in kwargs:
            kwargs['export_filename_prefix'] = self.filename_prefix
        self.data = self.do_analysis(**kwargs)

    @staticmethod
    def do_analysis(**kwargs):
        """
        Keyword Arguments:
            dataloader (callable): a callable which returns an sklean.base.Bunch
            export_filename_prefix (str): must be specified
            export_dirpath (str): default: dirpath(__file__) / 'pipelines'
            export_filename (str): default: export_filename_prefix + '_.py'
            export_filepath (str): default: export_dirpath / export_filename
            train_size (float): default: 0.75
            test_size (float): default: 0.25
            generations (int): default: 5
            population_size (int): default: 20
            verbosity (int): default: 2

        Returns:
            OrderedDict: dict of parameters
        """
        data = odict()
        export_filename_prefix = kwargs.pop('export_filename_prefix')
        export_dirpath = kwargs.pop('export_dirpath',
                                join(dirname(__file__), 'pipelines'))
        export_filename = kwargs.pop('export_filename',
                                     "%s_.py" % export_filename_prefix)
        export_filepath = kwargs.pop('export_filepath',
                                    join(export_dirpath, export_filename))
        data['export_filepath'] = export_filepath
        _export_dirpath = dirname(export_filepath)
        if not os.path.exists(_export_dirpath):
            os.makedirs(_export_dirpath)

        dataloader = kwargs['dataloader']
        data['dataloader'] = getattr(dataloader, '__qualname__',
                                     getattr(dataloader, '__name__',
                                             str(dataloader)))
        databunch = dataloader()

        tts_kwargs = odict()
        tts_kwargs['train_size'] = kwargs.pop('train_size', 0.75)
        tts_kwargs['test_size'] = kwargs.pop('test_size', 0.25)
        data.update(tts_kwargs)
        X_train, X_test, y_train, y_test = train_test_split(
            databunch.data, databunch.target, **tts_kwargs)

        regressor_kwargs = odict()
        regressor_kwargs['generations'] = kwargs.pop('generations', 5)
        regressor_kwargs['population_size'] = kwargs.pop('population_size', 20)
        regressor_kwargs['verbosity'] = kwargs.pop('verbosity', 2)
        data.update(regressor_kwargs)
        tpot = TPOTRegressor(**regressor_kwargs)

        log.info(TPOTAnalysis._to_json_str(data))
        tpot.fit(X_train, y_train)
        data['score'] = tpot.score(X_test, y_test)
        log.info(('score', data['score']))

        tpot.export(export_filepath)

        json_str = TPOTAnalysis._to_json_str(data)
        log.info(json_str)
        data['export_filepath_datajson'] = export_filepath + '.json'
        with open(data['export_filepath_datajson'], 'w') as f:
            f.write(json_str)

        return data

    @staticmethod
    def _to_json_str(data, indent=2):
        return json.dumps(data, indent=indent)

    def __str__(self):
        return self.to_json_str()


from functools import wraps
import data as _data

class HousePricesAnalysis(TPOTAnalysis):
    FILENAME_PREFIX = 'tpot_house_prices'

    @staticmethod
    def dataloader():
        return _data.load_house_prices()


def main():
    logging.basicConfig(level=logging.DEBUG)
    analysis = HousePricesAnalysis()
    data = analysis()
    print(data)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

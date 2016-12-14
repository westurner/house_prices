#!/usr/bin/env python
"""
| Src: https://rhiever.github.io/tpot/examples/Boston_Example/
| Src: https://github.com/rhiever/tpot/blob/master/docs/sources/examples/Boston_Example.md
| License: GPLv3 https://github.com/rhiever/tpot/blob/master/LICENSE
"""

from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

digits = load_boston()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')

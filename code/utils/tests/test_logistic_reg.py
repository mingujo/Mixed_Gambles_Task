"""test_diagnostics.py
Tests for the functions in logistic_reg.py module 

Run with:

    nosetests test_logistic_reg.py
"""


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
from logistic_reg import *
from organize_behavior_data import *
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_equal
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from scipy import stats
from statsmodels.formula.api import logit, ols

def test_add_gainlossratio():
	"""tests whether the gain/loss ratio is properly added in the data frame
	"""




def test_organize_columns():



def test_log_regression():






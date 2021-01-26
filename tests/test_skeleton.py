# -*- coding: utf-8 -*-

import pytest
from annette.skeleton import fib

__author__ = "mwessley"
__copyright__ = "mwessley"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)

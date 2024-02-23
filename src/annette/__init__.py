# -*- coding: utf-8 -*-
import os
from pathlib import Path
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


_DATABASE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)),'..','..')

def get_database(*args):
    print(args)
    return Path(_DATABASE_ROOT, 'database', *args)
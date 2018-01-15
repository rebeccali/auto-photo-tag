#!/usr/bin/env python3

'''This script runs the tagger
'''

import argparse
import sys
import places365 as pl
from datetime import datetime

parser = argparse.ArgumentParser(description='Run image tagger.')
parser.add_argument('filename', type=str, help='input image file')

args = parser.parse_args()

startTime = datetime.now()
imgplaces = pl.Place365(args.filename, True)

print('Process time: %0.2f microseconds' % (datetime.now() - startTime).microseconds )

imgplaces.print_identification()
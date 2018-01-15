#!/usr/bin/env python3

'''This script runs the tagger
'''

import argparse
import sys
import places365 as pl

parser = argparse.ArgumentParser(description='Run image tagger.')
parser.add_argument('filename', type=str, help='input image file')

args = parser.parse_args()

imgplaces = pl.Place365(args.filename, True)

print(imgplaces._env)
print(imgplaces._probs)
print(imgplaces._attributes)
print(imgplaces._categories)
imgplaces.print_identification()
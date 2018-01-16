#!/usr/bin/env python3

'''This script runs the tagger
'''

import argparse
import sys
import os
import glob
from datetime import datetime

import places365 as pl
import tagger

parser = argparse.ArgumentParser(description='Run image tagger.')
parser.add_argument('path', type=str, help='input image file')
parser.add_argument('--stfu', help='stfu', action='store_true', default=False)
parser.add_argument('--names-only', help='print filenames only', default=False, action='store_true' )

args = parser.parse_args()

acceptedFileExtentions = ['jpg','png', 'tif', 'dng', 'nef' ]

# get files in directory
if os.path.isfile(args.path):
    raw_paths = [os.path.abspath(args.path)]
elif os.path.isdir(args.path):
    raw_paths = sorted(glob.glob(args.path + '/*'))
else:
    raise Exception(args.path + ' is not a file or directory')

paths = []

for raw_path in raw_paths:
    if raw_path.lower().endswith(tuple(acceptedFileExtentions)):
        paths.append(raw_path)

num_images = len(paths)

if num_images == 0:
    raise Exception(args.path + ' contains no pictures. Accepted filetypes are '
        + ','.join(acceptedFileExtentions) + '.')


startTime = datetime.now()

for path in paths:
    imgplaces = pl.Place365(path, True)
    if args.names_only:
        print(imgplaces._imgpath)
    elif not args.stfu:
        imgplaces.print_identification()

elapsed_time = (datetime.now() - startTime).total_seconds()
elapsed_time_per_image = elapsed_time / num_images

if not args.stfu:
    print('Process time: %0.2f seconds' % elapsed_time )
    print('Process time per image: %0.2f milliseconds' % elapsed_time_per_image )


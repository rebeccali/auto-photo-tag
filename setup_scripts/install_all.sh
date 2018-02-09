#!/usr/bin/env bash

# This script tries to install everything

# exit on errors
set -e

# packages
./install_packages.sh

# avl
# This fails for damon, skip it for now.
#./build_avl.sh

# haskell
./install_conda.sh
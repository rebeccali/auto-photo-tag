#!/usr/bin/env bash

# This script installs all necessary apt packages

# exit on errors
set -e

sudo apt update

APT_PKGS="libexempi3"

sudo apt-get install -y ${APT_PKGS}
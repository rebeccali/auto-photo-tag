# Auto Photo Tag

Copyright (c) Rebecca Li 2018 under the MIT License.

Auto Photo Tag is a photo tagging tool that utilizes CNNs in order to identify scene attributes and the humans in the image, and populates those tags in the image metadata. Auto Photo Tag works on all kinds of images, including full size RAW image formats (TIF, DNG, NEF) as well as compressed image formats (JPEG, JPEG 2000, PNG). If a file does not support built in XMP metadata, a sidecar file will be created.

Auto Photo Tag utilizes the following CNN projects using PyTorch:
 * [Places Project](http://places2.csail.mit.edu)

## Dependencies

Unfortunately, this project relies upon many python 3 packages:

* Scipy
* OpenCV
* PIL
* Torch
* Numpy
* Python XMP toolkit

You can attempt to install the environment by running  `conda install --file conda.txt` in your conda environment. I try to keep `conda.txt` updated, but no guarantees.

There are alos some package dependencies:

* libexempi3

You should be able to run then by running `cd setup_scripts && ./install_all.sh`, but this is totally untested.

## Usage

Run `run_tagger.py` with an argument of a folder or file to be tagged.

Example:
```
./run_tagger.py testImages
```


## Features yet to be implemented

* Add a UI
* Add Human identifiers
* Add support for sidecar xmp

## Reference
Link: [Places2 Database](http://places2.csail.mit.edu), [Places1 Database](http://places.csail.mit.edu)

### Acknowledgements and License

Places dataset development has been partly supported by the National Science Foundation CISE directorate (#1016862), the McGovern Institute Neurotechnology Program (MINT), ONR MURI N000141010933, MIT Big Data Initiative at CSAIL, and Google, Xerox, Amazon and NVIDIA. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation and other funding agencies.

The pretrained places-CNN models can be used under the Creative Common License (Attribution CC BY). Please give appropriate credit, such as providing a link to our paper or to the [Places Project Page](http://places2.csail.mit.edu). The copyright of all the images belongs to the image owners.

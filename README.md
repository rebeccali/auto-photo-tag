# Auto Photo Tag

Copyright (c) Rebecca Li 2018 under the MIT License.

Auto Photo Tag is a photo tagging tool that utilizes CNNs in order to identify scene attributes and the humans in the image, and populates those tags in the image metadata. Auto Photo Tag works on all kinds of images, including full size RAW image formats (DNG, NEF, RAW) as well as compressed image formats (JPG, PNG, BMP, GIF, TIFF).

Auto Photo Tag utilizes the following CNN projects using PyTorch:
 * [Places Project](http://places2.csail.mit.edu)

## Dependencies

Unfortunately, this project relies upon many python 3 packages:

* Scipy
* OpenCV
* PIL
* Torch
* Numpy

## Usage

Run `run_tagger.py` with an argument of a folder or file to be tagged.

Example:
```
./run_tagger.py testImages
```


## Todo

* Add image converter
* Add tags to original raw images.
* Work on a folder of images
* Add a UI
* Add Human identifiers


## Reference
Link: [Places2 Database](http://places2.csail.mit.edu), [Places1 Database](http://places.csail.mit.edu)

### Acknowledgements and License

Places dataset development has been partly supported by the National Science Foundation CISE directorate (#1016862), the McGovern Institute Neurotechnology Program (MINT), ONR MURI N000141010933, MIT Big Data Initiative at CSAIL, and Google, Xerox, Amazon and NVIDIA. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation and other funding agencies.

The pretrained places-CNN models can be used under the Creative Common License (Attribution CC BY). Please give appropriate credit, such as providing a link to our paper or to the [Places Project Page](http://places2.csail.mit.edu). The copyright of all the images belongs to the image owners.

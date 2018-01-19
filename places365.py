#!/usr/bin/env python3

'''
    PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
    by Bolei Zhou, sep 2, 2017
'''

import cv2
from libxmp import XMPFiles, consts
import numpy as np
import os
from PIL import Image
from scipy.misc import imresize as imresize
import sys

import torch
from torch.autograd import Variable as V
from torch.nn import functional as F

import torchvision.models as models
from torchvision import transforms as trn


def image_var_laplacian(img):
    """calculates laplacian variance of image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def image_var_canny(img):
    """calculates canny edges variance"""
    return cv2.Canny(img,100,200).var()

def load_labels():
    """load places365 labels"""
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        print('categories_places365.txt not found!')
        sys.exit(1)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        print('IO_places365.txt not found!')
        sys.exit(1)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        print('labels_sunattribute.txt not found!')
        sys.exit(1)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        print('W_sceneattribute_wideresnet18.npy not found!')
        sys.exit(1)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def returnCAM(feature_conv, weight_softmax, class_idx):
    """create cam image"""
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for _ in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(imresize(cam_img, size_upsample))
    return output_cam

def returnTF():
    """load the image transformer"""
    tf = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

def compute_blurriness(imgpath):
    """ computes image blurriness using canny"""

    img_cv2 = cv2.imread(imgpath)
    # bluriness = self.image_var_laplacian(img_cv2)
    bluriness = image_var_canny(img_cv2)
    return bluriness

def update_xmp(imgpath):
    """ updates the xmp data in the image, or creates a sidecar xmp """
    embeddedXmpFormats = ['jpg', 'png', 'tif', 'gif', 'pdf']

    print(imgpath)

    xmpfile = XMPFiles( file_path=imgpath, open_forupdate=True)
    xmp = xmpfile.get_xmp()
    print(xmp.get_property(consts.XMP_NS_DC, 'format' ))
    return 0

class TaggedImage(object):
    '''
    imgpath is the path to the image to be processed
    render_cam is to generate a cam image
    output: environment (string), category probabilities (float, array),
    categories (string, array), attributes (string)
    '''
    features_blobs = []

    def get_imgpath(self):
        """gets imgpath"""
        return self._imgpath

    def hook_feature(self, module, input, output):
        """hooks feature for register hook"""
        self.features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    def print_identification(self):
        """prints out information on Places365"""
        print('----------------------------------------')

        print(self._imgpath)
        print('--BLURRY: ' + str(self._bluriness))
        if self._bluriness < 1000:
            print('I think it\'s blurry!')

        print('--TYPE OF ENVIRONMENT: ' + self._env)
        print('--SCENE CATEGORIES:')
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(self._probs[i], self._categories[i]))
        print('--SCENE ATTRIBUTES:')
        print(', '.join(self._attributes))

    def load_model(self):
        """Load places365 model """
        # this model has a last conv feature map as 14x14

        model_file = 'whole_wideresnet18_places365_python36.pth.tar'
        if not os.access(model_file, os.W_OK):
            print('whole_wideresnet18_places365_python36.pth.tar not found!')
            sys.exit(1)
        useGPU = 0
        if useGPU == 1:
            model = torch.load(model_file)
        else:
            model = torch.load(model_file, map_location=lambda storage, loc: storage) # allow cpu

        # the following is deprecated, everything is migrated to python36

        ## if you encounter the UnicodeDecodeError when use python3 to load the model,
        # add the following line will fix it. Thanks to @soravux
        #from functools import partial
        #import pickle
        #pickle.load = partial(pickle.load, encoding="latin1")
        #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

        model.eval()
        # hook the feature extractor
        features_names = ['layer4', 'avgpool'] # this is the last conv layer of the resnet
        for name in features_names:
            model._modules.get(name).register_forward_hook(self.hook_feature)
        return model

    def __init__(self, imgpath, render_cam=False):
        # load the labels
        classes, labels_IO, labels_attribute, W_attribute = load_labels()

        # self._classes = classes
        # load the model
        model = self.load_model()

        # load the transformer
        tf = returnTF() # image transformer

        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = params[-2].data.numpy()
        weight_softmax[weight_softmax<0] = 0

        # load the test image
        img = Image.open(imgpath)

        # this can only handle rgb images
        if not img.mode == 'RGB':
            img = img.convert('RGB')
        input_img = V(tf(img).unsqueeze(0), volatile=True)

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # output the IO prediction
        io_image = np.mean(labels_IO[idx[:10].numpy()]) # vote for the indoor or outdoor
        scene_env = ''
        if io_image < 0.5:
            scene_env = 'indoor'
        else:
            scene_env = 'outdoor'

        # output the scene attributes
        responses_attribute = W_attribute.dot(self.features_blobs[1])
        idx_a = np.argsort(responses_attribute)

        if render_cam:
            # generate class activation mapping
            CAMs = returnCAM(self.features_blobs[0], weight_softmax, [idx[0]])

            # render the CAM and output
            img = cv2.imread(imgpath)
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.4 + img * 0.5
            cv2.imwrite('cam.jpg', result)

        self._imgpath = imgpath
        self._bluriness = compute_blurriness(imgpath)
        self._probs = probs[:5]
        self._idx = probs[:5]
        self._env = scene_env
        self._categories = [classes[x] for x in idx[:5]]
        self._attributes = [labels_attribute[idx_a[i]] for i in range(-1, -10, -1)]

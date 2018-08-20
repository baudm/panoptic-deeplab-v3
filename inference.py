"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import deeplab_model
from utils import preprocessing
from utils import dataset_util
from mapping import Maps as mapLU

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as sio
import skimage.color as sco

from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='dataset/VOCdevkit/VOC2012/JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='./dataset/inference_output',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='./dataset/sample_images_list.txt',
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 54
STUFF_CLEANING = False

#create the object to call the maps
cat_map = mapLU()

def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_plus_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
      })

  examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
  image_files = [os.path.join(FLAGS.data_dir, filename) for filename in examples]

  predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files),
        hooks=pred_hooks)

  output_dir = FLAGS.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)


  gt_path = '/media/airscan/Disk3/MS_COCO/segmentations2'
  for pred_dict, image_path in zip(predictions, image_files):
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = image_basename + '_mask.png'
    output_filename2 = image_basename + '.png' #needed filename for evaluation of panoptic
    gt_filename = os.path.join(gt_path,image_basename+'.png')

    # create 3 output folders (1) colored (2) stuff classes (3) 2-ch output
    output_dir_colored = os.path.join(output_dir,'colored')
    output_dir_segmask = os.path.join(output_dir,'masks')
    output_dir_ch2mask = os.path.join(output_dir,'ch2_outputs')

    path_to_output_colored = os.path.join(output_dir_colored, output_filename)
    path_to_output_segmask = os.path.join(output_dir_segmask, output_filename2)
    path_to_output_ch2mask = os.path.join(output_dir_ch2mask, output_filename2)

    print("generating:", image_basename + ' segmentations')

    if not STUFF_CLEANING:
        mask = pred_dict['decoded_labels'] # output is the colormap segmentation
        mask = Image.fromarray(mask.astype(np.uint8))
        mask.save(path_to_output_colored)

    # returns the panoptic classes
    segmask1 = pred_dict['classes']
    segmask1 = np.squeeze(segmask1)
    segmask1 = np.vectorize(cat_map.stuff_category_rev.get)(segmask1)

    # load the preprocessing by the confusion matrix here
    conf_mat = np.load('conf-matrix.npz')['cm']

    # with confusion matrix
    segmask_prob = pred_dict['probabilities']
    segmask_new_prob = np.dot(segmask_prob,conf_mat)
    segmask = np.argmax(segmask_new_prob,axis=-1)
    segmask = np.vectorize(cat_map.stuff_category_rev.get)(segmask)

    # for direct output of deeplab

    segmask1_PIL = Image.fromarray(segmask1.astype(np.uint8))
    segmask1_PIL.save(path_to_output_segmask)

    if not STUFF_CLEANING: #used only for creating panoptic outputs
        ch2mask = np.zeros((*segmask.shape,3))
        ch2mask[:,:,0] = segmask
        ch2mask = Image.fromarray(ch2mask.astype(np.uint8))
        ch2mask.save(path_to_output_ch2mask)

    """
    # horz cat for visual comparison
    input = sio.imread(image_path)
    gt = sio.imread(gt_filename)
    gt = sco.gray2rgb(gt)
    segmask = sco.gray2rgb(segmask)
    segmask1 = sco.gray2rgb(segmask1)

    comb1 = np.hstack([input,gt])
    comb2 = np.hstack([segmask1,segmask])
    comb2 = np.vectorize(cat_map.stuff_category_mapping.get)(comb2)
    comb = np.vstack([comb1,comb2])
    comb_PIL = Image.fromarray(comb.astype(np.uint8))
    comb_PIL.save(os.path.join(output_dir,'compare',output_filename))
    """

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

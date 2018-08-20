"""Utility functions for preprocessing data sets."""

from PIL import Image
import numpy as np
import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

# colour map
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

# colour map of coco panoptic segmentation
label_colours_pan = [(0,0,0), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192),
                     (250, 170, 30), (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
                     (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92), (209, 0, 151),
                     (188, 208, 182), (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                     (134, 134, 103), (145, 148, 174), (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255), (151, 0, 95), (9, 80, 61),
                     (84, 105, 51), (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
                     (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
                     (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15), (127, 167, 115),
                     (59, 105, 106), (142, 108, 45), (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122), (191, 162, 208), (255, 255, 128),
                     (147, 211, 203), (150, 100, 100), (168, 171, 172), (146, 112, 198), (210, 170, 100),(92, 136, 89),  (218, 88, 184), (241, 129, 0),
                     (217, 17, 255),  (124, 74, 181),  (70, 70, 70),    (255, 228, 255), (154, 208, 0),  (193, 0, 92),   (76, 91, 113),  (255, 180, 195), (106, 154, 176),
                     (230, 150, 140), (60, 143, 255),  (128, 64, 128),  (92, 82, 55),    (254, 212, 124),(73, 77, 174),  (255, 160, 98), (255, 255, 255), (104, 84, 109),
                     (169, 164, 131), (225, 199, 255), (137, 54, 74),   (135, 158, 223), (7, 246, 231),  (107, 255, 200),(58, 41, 149),  (183, 121, 142), (255, 73, 97),
                     (107, 142, 35),  (190, 153, 153), (146, 139, 141), (70, 130, 180),  (134, 199, 156),(209, 226, 140),(96, 36, 108),  (96, 96, 96),   (64, 170, 64),
                     (152, 251, 152), (208, 229, 228), (206, 186, 171), (152, 161, 64),  (116, 112, 0),  (0, 114, 143),  (102, 102, 156),(250, 141, 255)]

panoptic_category_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15,
                            17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29,
                            34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43,
                            49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57,
                            63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71,
                            81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80, 92: 81, 93: 82, 95: 83, 100: 84, 107: 85,
                            109: 86, 112: 87, 118: 88, 119: 89, 122: 90, 125: 91, 128: 92, 130: 93, 133: 94, 138: 95, 141: 96, 144: 97, 145: 98,
                            147: 99, 148: 100, 149: 101, 151: 102, 154: 103, 155: 104, 156: 105, 159: 106, 161: 107, 166: 108, 168: 109,
                            171: 110, 175: 111, 176: 112, 177: 113, 178: 114, 180: 115, 181: 116, 184: 117, 185: 118, 186: 119, 187: 120, 188: 121,
                            189: 122, 190: 123, 191: 124, 192: 125, 193: 126, 194: 127, 195: 128, 196: 129, 197: 130, 198: 131, 199: 132, 200: 133}

stuff_category_mapping = {0: 0, 92: 1, 93: 2, 95: 3, 100: 4, 107: 5, 109: 6, 112: 7, 118: 8, 119: 9, 122: 10,
                          125: 11, 128: 12, 130: 13, 133: 14, 138: 15, 141: 16, 144: 17, 145: 18, 147: 19, 148: 20,
                          149: 21, 151: 22, 154: 23, 155: 24, 156: 25, 159: 26, 161: 27, 166: 28, 168: 29, 171: 30,
                          175: 31, 176: 32, 177: 33, 178: 34, 180: 35, 181: 36, 184: 37, 185: 38, 186: 39, 187: 40,
                          188: 41, 189: 42, 190: 43, 191: 44, 192: 45, 193: 46, 194: 47, 195: 48, 196: 49, 197: 50,
                          198: 51, 199: 52, 200: 53}

label_colours_stuff = [(0,0,0),(191, 162, 208), (255, 255, 128), (147, 211, 203), (150, 100, 100), (168, 171, 172), (146, 112, 198), (210, 170, 100),
                       (92, 136, 89),  (218, 88, 184), (241, 129, 0), (217, 17, 255),  (124, 74, 181),  (70, 70, 70),    (255, 228, 255), (154, 208, 0),
                       (193, 0, 92),   (76, 91, 113),  (255, 180, 195), (106, 154, 176), (230, 150, 140), (60, 143, 255),  (128, 64, 128),  (92, 82, 55),
                       (254, 212, 124),(73, 77, 174),  (255, 160, 98), (255, 255, 255), (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
                       (135, 158, 223), (7, 246, 231),  (107, 255, 200),(58, 41, 149),  (183, 121, 142), (255, 73, 97), (107, 142, 35),  (190, 153, 153),
                       (146, 139, 141), (70, 130, 180),  (134, 199, 156),(209, 226, 140),(96, 36, 108),  (96, 96, 96),   (64, 170, 64), (152, 251, 152),
                       (208, 229, 228), (206, 186, 171), (152, 161, 64),  (116, 112, 0),  (0, 114, 143),  (102, 102, 156),(250, 141, 255)]


def decode_labels(mask, num_images=1, num_classes=21):
  """Decode batch of segmentation masks.

  Args:
    mask: result of inference after taking argmax.
    num_images: number of images to decode from the batch.
    num_classes: number of classes to predict (including background).

  Returns:
    A batch with num_images RGB images of the same size as the input.
  """
  n, h, w, c = mask.shape
  assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
                            % (n, num_images)
  outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
  for i in range(num_images):
    img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
    pixels = img.load()
    for j_, j in enumerate(mask[i, :, :, 0]):
      for k_, k in enumerate(j):
        if k < num_classes:
          pixels[k_, j_] = label_colours[k]
    outputs[i] = np.array(img)
  return outputs

def decode_labels_pan(mask, num_images=1, num_classes=21):
  """Decode batch of segmentation masks.

  Args:
    mask: result of inference after taking argmax.
    num_images: number of images to decode from the batch.
    num_classes: number of classes to predict (including background).

  Returns:
    A batch with num_images RGB images of the same size as the input.
  """

  n, h, w, c = mask.shape
  assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
                            % (n, num_images)
  outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
  for i in range(num_images):
    img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
    pixels = img.load()
    for j_, j in enumerate(mask[i, :, :, 0]):
      for k_, k in enumerate(j):
        """
        if k not in panoptic_category_mapping:
            while k not in panoptic_category_mapping:
                if k > 200:
                    k -= 1
                else:
                    k += 1
        """
        #if k not in panoptic_category_mapping:
        #    kNew = 0 # make it void
        #else:
        #    kNew = panoptic_category_mapping[k]
        if k < num_classes:
          pixels[k_, j_] = label_colours_stuff[k]
    outputs[i] = np.array(img)
  return outputs

def mean_image_addition(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
  """Adds the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] += means[i]
  return tf.concat(axis=2, values=channels)


def mean_image_subtraction(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def random_rescale_image_and_label(image, label, min_scale, max_scale):
  """Rescale an image and label with in target scale.

  Rescales an image and label within the range of target scale.

  Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[height, width, 1]`.
    min_scale: Min target scale.
    max_scale: Max target scale.

  Returns:
    Cropped and/or padded image.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
    If `labels` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, 1]`.
  """
  if min_scale <= 0:
    raise ValueError('\'min_scale\' must be greater than 0.')
  elif max_scale <= 0:
    raise ValueError('\'max_scale\' must be greater than 0.')
  elif min_scale >= max_scale:
    raise ValueError('\'max_scale\' must be greater than \'min_scale\'.')

  shape = tf.shape(image)
  height = tf.to_float(shape[0])
  width = tf.to_float(shape[1])
  scale = tf.random_uniform(
      [], minval=min_scale, maxval=max_scale, dtype=tf.float32)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  image = tf.image.resize_images(image, [new_height, new_width],
                                 method=tf.image.ResizeMethod.BILINEAR)
  # Since label classes are integers, nearest neighbor need to be used.
  label = tf.image.resize_images(label, [new_height, new_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return image, label


def random_crop_or_pad_image_and_label(image, label, crop_height, crop_width, ignore_label):
  """Crops and/or pads an image to a target width and height.

  Resizes an image to a target width and height by rondomly
  cropping the image or padding it evenly with zeros.

  Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[height, width, 1]`.
    crop_height: The new height.
    crop_width: The new width.
    ignore_label: Label class to be ignored.

  Returns:
    Cropped and/or padded image.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """
  label = label - ignore_label  # Subtract due to 0 padding.
  label = tf.to_float(label)
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  image_and_label = tf.concat([image, label], axis=2)
  image_and_label_pad = tf.image.pad_to_bounding_box(
      image_and_label, 0, 0,
      tf.maximum(crop_height, image_height),
      tf.maximum(crop_width, image_width))
  image_and_label_crop = tf.random_crop(
      image_and_label_pad, [crop_height, crop_width, 4])

  image_crop = image_and_label_crop[:, :, :3]
  label_crop = image_and_label_crop[:, :, 3:]
  label_crop += ignore_label
  label_crop = tf.to_int32(label_crop)

  return image_crop, label_crop


def random_flip_left_right_image_and_label(image, label):
  """Randomly flip an image and label horizontally (left to right).

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    label: A 3-D tensor of shape `[height, width, 1].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
  """
  uniform_random = tf.random_uniform([], 0, 1.0)
  mirror_cond = tf.less(uniform_random, .5)
  image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
  label = tf.cond(mirror_cond, lambda: tf.reverse(label, [1]), lambda: label)

  return image, label


def eval_input_fn(image_filenames, label_filenames=None, batch_size=1):
  """An input function for evaluation and inference.

  Args:
    image_filenames: The file names for the inferred images.
    label_filenames: The file names for the grand truth labels.
    batch_size: The number of samples per batch. Need to be 1
        for the images of different sizes.

  Returns:
    A tuple of images and labels.
  """
  # Reads an image from a file, decodes it into a dense tensor
  def _parse_function(filename, is_label):
    if not is_label:
      image_filename, label_filename = filename, None
    else:
      image_filename, label_filename = filename

    image_string = tf.read_file(image_filename)
    image = tf.image.decode_image(image_string)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    image = mean_image_subtraction(image)

    if not is_label:
      return image
    else:
      label_string = tf.read_file(label_filename)
      label = tf.image.decode_image(label_string)
      label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
      label.set_shape([None, None, 1])

      return image, label

  if label_filenames is None:
    input_filenames = image_filenames
  else:
    input_filenames = (image_filenames, label_filenames)

  dataset = tf.data.Dataset.from_tensor_slices(input_filenames)
  if label_filenames is None:
    dataset = dataset.map(lambda x: _parse_function(x, False))
  else:
    dataset = dataset.map(lambda x, y: _parse_function((x, y), True))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()

  if label_filenames is None:
    images = iterator.get_next()
    labels = None
  else:
    images, labels = iterator.get_next()

  return images, labels

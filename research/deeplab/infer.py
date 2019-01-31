# Run inference on all jpg files in a given folder and output annotated images in an output folder.

#@title Imports

from datetime import datetime
import os
from io import BytesIO
import tarfile
import tempfile

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

#@title Helper methods

_DPI = 72
_FIG_SIZE = (12.0138, 6.75)

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 1280
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, trained_model_dir):
    """Loads pretrained deeplab model."""
    self.graph = tf.Graph()

    file_name = trained_model_dir + self.FROZEN_GRAPH_NAME + '.pb'
    #file_name = '/research/deeplab/trained_models/deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb'
        
    with open(file_name, 'rb') as f:
      graph_def = tf.GraphDef.FromString(f.read())

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image, verbose=False):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    if verbose: print("image size {}x{}".format(width, height))
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    if verbose: print("target size {}".format(target_size))
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [128, 64, 128],
      [244, 35, 232],
      [70, 70, 70],
      [102, 102, 156],
      [190, 153, 153],
      [153, 153, 153],
      [250, 170, 30],
      [220, 220, 0],
      [107, 142, 35],
      [152, 251, 152],
      [70, 130, 180],
      [220, 20, 60],
      [255, 0, 0],
      [0, 0, 142],
      [0, 0, 70],
      [0, 60, 100],
      [0, 80, 100],
      [0, 0, 230],
      [119, 11, 32],
  ])


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_cityscapes_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(36, 12))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


# Cityscapes labels
LABEL_NAMES = np.asarray([
    'road', 
    'sidewalk', 
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
])
print('Number of labels: {}'.format(len(LABEL_NAMES)))


FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

#@title Select and download models {display-mode: "form"}

MODEL_NAME = 'xception71_dpc_cityscapes_trainfine'  # @param ['mobilenetv2_coco_cityscapes_trainfine', 'xception65_cityscapes_trainfine', 'xception71_dpc_cityscapes_trainfine', 'xception71_dpc_cityscapes_trainval']

'''
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_cityscapes_trainfine':
        'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz',
    'xception65_cityscapes_trainfine':
        'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
    'xception71_dpc_cityscapes_trainfine':
        'deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz',
    'xception71_dpc_cityscapes_trainval':
        'deeplab_cityscapes_xception71_trainvalfine_2018_09_08.tar.gz',    
}
'''

# Extract above pre-trained model tar.gz files to the following folder.
#_PROJECT_DIR = '/home/ritpower/rick/deeplab/trained_models/'
_PROJECT_DIR = '/research/deeplab/trained_models/'
_MODEL_DIR = {
    'mobilenetv2_coco_cityscapes_trainfine':
        'deeplabv3_mnv2_cityscapes_train',
    'xception65_cityscapes_trainfine':
        'deeplabv3_cityscapes_train',
    'xception71_dpc_cityscapes_trainfine':
        'train_fine',
    'xception71_dpc_cityscapes_trainval':
        'trainval_fine',
}

model_path = _PROJECT_DIR + _MODEL_DIR[MODEL_NAME] + '/'
print('model_path: {}'.format(model_path))

MODEL = DeepLabModel(model_path)
print('model loaded successfully!')

def save_vis(image, seg_map, file_name):
  print("type(image) {}".format(type(image)))
  print("type(seg_map) {}".format(type(seg_map)))
  print("seg_map.shape {}".format(seg_map.shape))
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  print("type(seg_image) {}".format(type(seg_image)))
  print("seg_image.shape {}".format(seg_image.shape))
  fig = plt.figure(figsize=_FIG_SIZE)
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.grid('off')
  print("Writing to {}".format(file_name))
  plt.savefig(file_name, dpi=_DPI, bbox_inches='tight')
  plt.close(fig)
    
  im = np.asarray(image)
  print("type(im) {}".format(type(im)))
  print("im.shape {}".format(im.shape))

def nparray_info(x):
  print("shape {} dtype {} max {} min {}".format(x.shape, x.dtype, np.amax(x), np.amin(x)))
    
def pillow_info(x):
  print("mode {} size {}".format(x.mode, x.size))
    
def save_vis_jpg(image, seg_map, file_name, verbose=False):
  if verbose: print("save_vis_jpg")
  if verbose: print("type(seg_map) {}".format(type(seg_map)))
  if verbose: nparray_info(seg_map)
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  if verbose: nparray_info(seg_image)
  seg_image = Image.fromarray(np.uint8(seg_image))
  if verbose: pillow_info(image)
  if verbose: pillow_info(seg_image)
  blended = Image.blend(image, seg_image, alpha=0.6)
  blended.save(file_name, 'JPEG', quality=80)


#@title Run on sample images {display-mode: "form"}

def run_visualization(file_name, out_file, verbose=False):
  """Inferences DeepLab model and visualizes result."""
  with open(file_name, 'rb') as f:
    jpeg_str = f.read()
    original_im = Image.open(BytesIO(jpeg_str))

  if verbose: print('running deeplab on image %s...' % file_name)
  if verbose: t0 = datetime.now()
  resized_im, seg_map = MODEL.run(original_im)
  if verbose: print("Infered in {} seconds".format((datetime.now()-t0).total_seconds()))

  if verbose: print('running visualization...')
  if verbose: vis_segmentation(resized_im, seg_map)
    
  #save_vis(resized_im, seg_map, out_file)
  save_vis_jpg(resized_im, seg_map, out_file)


INPUT_DIR = '/research/deeplab/data/input/'
OUTPUT_DIR = '/research/deeplab/data/output/'
import glob
all_files = glob.glob(INPUT_DIR + "*.jpg")
nb_files = len(all_files)
print("Number of input files: {}".format(nb_files))

if nb_files:
  print(all_files[0])
  print(os.path.basename(all_files[0]))
  print(os.path.dirname(all_files[0]))

  for f in all_files:
    out_f = OUTPUT_DIR + os.path.basename(f)
    print(out_f)
    run_visualization(f, out_f)


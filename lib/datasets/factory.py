# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
#from datasets.coco import coco
from datasets.vg import vg
from datasets.clevr import clevr

import numpy as np

# Set up vg_<split>
for split in ['train', 'validation', 'test']:
  name = 'visual_genome_{}'.format(split)
  __sets[name] = (lambda split=split: vg(split))

# Set up clevr_<split>
for split in ['train', 'val', 'test']:
  name = 'clevr_{}'.format(split)
  __sets[name] = (lambda split=split: clevr(split))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())

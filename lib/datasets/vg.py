# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append("../..")

from datasets import models
sys.modules['Data.VisualGenome.models'] = models

import os

import cPickle

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from model.config import cfg


class vg(imdb):
  def __init__(self, image_set, use_diff=False):
    name = 'visual_genome'
    if use_diff:
      name += '_diff'
    imdb.__init__(self, name)
    self._image_set = image_set
    self._devkit_path = self._get_default_path()
    self._data_path = self._devkit_path
    self._classes = ["invalid", "subject", "object", "none"]
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._data = self._load_data()
    self._image_index = [i for i in self._data if hasattr(i, 'queries_gt')]
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': use_diff,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    url = index.image.url
    path_lst = url.split('/')

    image_path = os.path.join(self._data_path, 'JPEGImages', path_lst[-2], path_lst[-1])
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path


  def _get_default_path(self):
    """
    Return the default path where vg is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'visual_genome')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file) and False:
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_pascal_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info
    """
    image = index

    num_objs = 0
    for ix, obj in enumerate(image.objects):
      if image.objects[ix].x < 0 or image.objects[ix].y < 0 or image.objects[ix].x + image.objects[ix].width > 1023 or \
                              image.objects[ix].y + image.objects[ix].height > 1023:
        continue
      num_objs += 1

    boxes = np.zeros((num_objs, 4), dtype=np.float32)


    gt_classes = np.zeros((0, num_objs, 1), dtype=np.int32)
    overlaps = np.zeros((0, num_objs, self.num_classes), dtype=np.int64)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    queries = np.zeros((0, 235), dtype=np.float32)
    # Load object bounding boxes into a data frame.
    index = 0
    for ix, obj in enumerate(image.objects):
      if image.objects[ix].x < 0 or image.objects[ix].y < 0 or image.objects[ix].x + image.objects[ix].width > 1023 or \
                              image.objects[ix].y + image.objects[ix].height > 1023:
        continue
      # Make pixel indexes 0-based
      x1_offset = 0.0#image.objects[ix].width * (-0.1)
      x2_offset = 0.0#image.objects[ix].width * 0.1
      y1_offset = 0.0#image.objects[ix].height * (-0.1)
      y2_offset = 0.0#image.objects[ix].height * 0.1
      boxes[index][0] = max((image.objects[ix].x + x1_offset), 0.0)
      boxes[index][1] = max((image.objects[ix].y + y1_offset), 0.0)
      boxes[index][2] = min((image.objects[ix].x + x2_offset + image.objects[ix].width), 1023.0)
      boxes[index][3] = min((image.objects[ix].y + y2_offset + image.objects[ix].height), 1023.0)
      seg_areas[index] = (boxes[index][2] - boxes[index][0] + 1.0) * (boxes[index][3] - boxes[index][1] + 1.0)
      index += 1
    #load gt classes
     
    for query_index in range(image.queries_gt.shape[0]):
      query_gt_classes = np.zeros((1, num_objs, 1), dtype=np.int32)
      query_overlaps = np.zeros((1, num_objs, self.num_classes), dtype=np.int64)
      query_overlaps[0, :, 3] = 1
      query_gt_classes[0, :, 0] = 3
      if image.one_hot_relations_gt[query_index][-1] == 1:
        # print "negative triplet"
        continue

      sub = image.one_hot_relations_gt[query_index][:96]
      obj = image.one_hot_relations_gt[query_index][96:96 * 2]
      rel = image.one_hot_relations_gt[query_index][96 * 2:]
      found = False
      i_index = 0
      for i in range(image.objects_labels.shape[0]):
        if image.objects[i].x < 0 or image.objects[i].y < 0 or image.objects[i].x + image.objects[i].width > 1023 or \
                                image.objects[i].y + image.objects[i].height > 1023:
          continue
        if not np.array_equal(image.objects_labels[i], sub):
          i_index += 1
          continue
        j_index = 0
        for j in range(image.objects_labels.shape[0]):
          if image.objects[j].x < 0 or image.objects[j].y < 0  or image.objects[j].x + image.objects[j].width > 1023 or image.objects[j].y + image.objects[j].height > 1023:
            continue

          if not np.array_equal(image.objects_labels[j], obj):
            j_index += 1
            continue
          if np.array_equal(rel, image.predicates_labels[i, j]):
            query_gt_classes[0, i_index, 0] = 1
            query_overlaps[0, i_index, 1] = 1
            query_overlaps[0, i_index, 3] = 0
            query_gt_classes[0, j_index, 0] = 2
            query_overlaps[0, j_index, 2] = 1
            query_overlaps[0, j_index, 3] = 0
            found = True
          j_index += 1
        i_index += 1
      if not found:
        continue
      gt_classes = np.concatenate((gt_classes, query_gt_classes), axis=0)
      overlaps = np.concatenate((overlaps, query_overlaps), axis=0)
      queries = np.concatenate((queries, image.one_hot_relations_gt[query_index].reshape([1,-1])), axis=0)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas,
             'query' : queries}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      'VOC' + self._year,
      'Main',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
      pass
  def _do_python_eval(self, output_dir='output'):
    pass

  def _do_matlab_eval(self, output_dir='output'):
      pass

  def evaluate_detections(self, all_boxes, output_dir):
      pass
  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True

  def _load_data(self):
    print("start loading data " + self._image_set)
    file_path = os.path.join(self._data_path, self._image_set + ".p")
    file_handle = open(file_path, "rb")
    images = cPickle.load(file_handle)
    file_handle.close()
    print("done loading data " + self._image_set)
    return images


if __name__ == '__main__':

  d = vg('train')
  #res = d.roidb

  from IPython import embed;

  embed()

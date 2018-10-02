# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from nets.gpi import Gpi
from utils.visualization import draw_bounding_boxes
from model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf

from model.config import cfg

class Network(object):
  def __init__(self):
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = []
    self._event_summaries = {}
    self._variables_to_fix = {}

  def _add_gt_image(self):
    # add back mean
    image = self._image + cfg.PIXEL_MEANS
    # BGR to RGB (opencv uses BGR)
    resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
    self._gt_image = tf.reverse(resized, axis=[-1])

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    if self._gt_image is None:
      self._add_gt_image()
    image = tf.py_func(draw_bounding_boxes, 
                      [self._gt_image, self._gt_boxes, self._gt_labels, self._im_info],
                      tf.float32, name="gt_boxes")
    
    return tf.summary.image('GROUND_TRUTH', image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if True or name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      if cfg.USE_E2E_TF:
        rois, rpn_scores, top_anchors = proposal_top_layer_tf(
          rpn_cls_prob,
          rpn_bbox_pred,
          self._im_info,
          self._feat_stride,
          self._anchors,
          self._num_anchors
        )
      else:
        rois, rpn_scores, top_anchors = tf.py_func(proposal_top_layer,
                                                   [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                                    self._feat_stride, self._anchors, self._num_anchors],
                                                   [tf.float32, tf.float32], name="proposal_top")

      rois.set_shape([-1, 5])
      rpn_scores.set_shape([-1, 1])

    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      if cfg.USE_E2E_TF:
        rois, rpn_scores, indices = proposal_layer_tf(
          rpn_cls_prob,
          rpn_bbox_pred,
          self._im_info,
          self._mode,
          self._feat_stride,
          self._anchors,
          self._num_anchors
        )
        self._predictions["indices"] = indices
      else:
        rois, rpn_scores = tf.py_func(proposal_layer,
                              [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                               self._feat_stride, self._anchors, self._num_anchors],
                              [tf.float32, tf.float32], name="proposal")

      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bounding boxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

      pw_x1, pw_y1, pw_x2, pw_y2 = self.union_box(x1, y1, x2, y2)

      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      pw_bboxes = tf.stop_gradient(tf.stack([pw_y1, pw_x1, pw_y2, pw_x2], axis=2))
      pre_pool_size = cfg.POOLING_SIZE * 2
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
      pw_crops = tf.image.crop_and_resize(bottom, pw_bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                       name="pw_crops")
      self._predictions['pred_bbox'] = tf.stop_gradient(tf.concat([x1, y1, x2, y2], axis=1))
      self._predictions['pw_pred_bbox'] = tf.stop_gradient(tf.concat([pw_x1, pw_y1, pw_x2, pw_y2], axis=1))
    return slim.max_pool2d(crops, [2, 2], padding='SAME'), slim.max_pool2d(pw_crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _anchor_target_layer(self, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32],
        name="anchor_target")

      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

      self._score_summaries.update(self._anchor_targets)

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores, name):
    with tf.variable_scope(name) as scope:
      rois, roi_scores, labels, partial_entity_class, partial_relation_class, bbox_targets, bbox_inside_weights, bbox_outside_weights, labels_mask = tf.py_func(
        proposal_target_layer,
        [rois, roi_scores, self._gt_boxes, self._gt_labels, self._partial_entity_class, self._partial_relation_class, self._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        name="proposal_target")

      rois.set_shape([None, 5])
      roi_scores.set_shape([None])
      labels.set_shape([self._gt_labels.shape[0], None, 1])
      bbox_targets.set_shape([None, 4])
      bbox_inside_weights.set_shape([None, 4])
      bbox_outside_weights.set_shape([None, 4])
      partial_entity_class.set_shape([None, 96])
      partial_relation_class.set_shape([None, None, 43])

      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['partial_entity_class'] = tf.to_int32(partial_entity_class, name="to_int32")
      self._proposal_targets['partial_relation_class'] = tf.to_int32(partial_relation_class, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
      self._proposal_targets['labels_mask'] = tf.stop_gradient(labels_mask)

      self._score_summaries.update(self._proposal_targets)

      return rois, roi_scores

  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
      if cfg.USE_E2E_TF:
        anchors, anchor_length = generate_anchors_pre_tf(
          height,
          width,
          self._feat_stride,
          self._anchor_scales,
          self._anchor_ratios
        )
      else:
        anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                            [height, width,
                                             self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                            [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  def _build_network(self, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    net_conv = self._image_to_head(is_training)
    with tf.variable_scope(self._scope, self._scope):
      # build the anchors for the image
      self._anchor_component()
      # region proposal network
      rois = self._region_proposal(net_conv, is_training, initializer)
      # region of interest pooling
      if cfg.POOLING_MODE == 'crop':
        pool5, pw_pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
        
        gt_rois = tf.concat((tf.zeros_like(self._gt_boxes[:,:1]), self._gt_boxes), axis=1)
        gt_pool5, gt_pw_pool5 = self._crop_pool_layer(net_conv, gt_rois, "gt_pool5")
      else:
        raise NotImplementedError
    self._predictions["pool5"] = pool5
    fc7, pw_fc7 = self._head_to_tail(pool5, pw_pool5, is_training)
    gt_fc7, gt_pw_fc7 = self._head_to_tail(gt_pool5, gt_pw_pool5, is_training, name="gt", reuse=True)
    #fc7 = pool5
    #pw_fc7 = pw_pool5
    #fc7 = tf.reduce_mean(fc7, axis=[1, 2])
    #pw_fc7 = tf.reduce_mean(pw_fc7, axis=[1, 2])

    # GPI
    fc7 = tf.concat((fc7, self._predictions['pred_bbox0_pool5']), axis=1)
    pw_fc7 = tf.concat((pw_fc7, self._predictions['pw_pred_bbox0_pool5']), axis=1)
    N = tf.slice(tf.shape(fc7), [0], [1])
    shape = tf.concat((N, tf.shape(fc7)), 0)
    pw_fc7 = tf.reshape(pw_fc7, shape)
    

    N = tf.slice(tf.shape(gt_fc7), [0], [1])
    shape = tf.concat((N, tf.shape(gt_fc7)), 0)
    gt_pw_fc7 = tf.reshape(gt_pw_fc7, shape)
    
    #fc7 = tf.to_float(self._proposal_targets['partial_entity_class'])
    #pw_fc7 = tf.to_float(self._proposal_targets['partial_relation_class'])
    #pw_fc7 = tf.concat((pw_fc7, tf.transpose(pw_fc7, perm=[1,0,2])), axis=2)

    self.gpi = Gpi()
    pred_node_features, ent_score, rel_score, ent_score0, rel_score0 = self.gpi.predict(fc7, pw_fc7, gt_fc7, gt_pw_fc7)
    #pred_node_features = fc7
    self._predictions['ent_cls_score'] = ent_score
    self._predictions['rel_cls_score'] = rel_score
    self._predictions['ent_cls_score0'] = ent_score0
    self._predictions['rel_cls_score0'] = rel_score0
    

    with tf.variable_scope(self._scope, self._scope):
      # region classification
      N = tf.slice(tf.shape(pred_node_features), [0], [1], name="N")
      expand_query_shape = tf.concat((N, tf.shape(self._query)), 0)
      expand_query = tf.zeros(expand_query_shape) + self._query
      expand_query = tf.transpose(expand_query, perm=[1,0,2])


      Q = tf.slice(tf.shape(self._query), [0], [1], name="Q")
      expand_node_features_shape = tf.concat((Q, tf.shape(pred_node_features)), 0)
      expand_node_features = tf.add(tf.zeros(expand_node_features_shape), pred_node_features)
            
      expand_node_features = tf.concat((expand_node_features, expand_query), axis=2)
      self.expand_node_features = expand_node_features
      cls_prob, bbox_pred = self._region_classification(pred_node_features, expand_node_features, is_training,
                                                      initializer, initializer_bbox)

    self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

  def _add_losses(self, sigma_rpn=3.0):
    with tf.variable_scope('LOSS_' + self._tag) as scope:
      # RPN, class loss
      rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
      rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
      rpn_select = tf.where(tf.not_equal(rpn_label, -1))
      rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
      rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
      rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

      # RPN, bbox loss
      rpn_bbox_pred = self._predictions['rpn_bbox_pred']
      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
      rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
      # mask
      mask = tf.abs(tf.reshape(self._proposal_targets['labels_mask'], (-1,1)))

      # RCNN, class loss
      Q = tf.slice(tf.shape(self._query), [0], [1], name="Q")
      expand_mask_shape = tf.concat((Q, tf.shape(mask)), 0)
      expand_mask = tf.add(tf.zeros(expand_mask_shape), mask)
      self.expand_mask = expand_mask
      label = self._proposal_targets["labels"]
      masked_label = tf.multiply(expand_mask, tf.to_float(label))
      nof_poss = (1.0 + tf.reduce_sum(tf.to_float(tf.equal(masked_label, 1) | tf.equal(masked_label, 2)), axis=1))
      nof_negs = (1.0 + tf.reduce_sum(tf.to_float(tf.equal(masked_label, 0) | tf.equal(masked_label, 3)), axis=1))
      factor = 3.0 * tf.to_float(nof_poss) / nof_negs
      self.factor = factor

      factor = tf.squeeze(factor)
      expand_mask = tf.squeeze(expand_mask)
      label = tf.squeeze(label)

      expand_mask_factor = tf.transpose(tf.transpose(expand_mask) * factor)
      
      expand_mask = tf.where(tf.equal(label, 0) | tf.equal(label, 3), expand_mask_factor, expand_mask)
      self.expand_mask2 = expand_mask
      expand_mask = tf.reshape(expand_mask, [-1])
      cls_score = tf.reshape(self._predictions["cls_score"], [-1, self._num_classes])
      label = tf.reshape(label, [-1])
      nof = tf.reduce_sum(expand_mask) + 1.0
      self.nof = nof
      ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)
      self.ce = ce
      w_ce = tf.multiply(expand_mask, ce)
      self.wce = w_ce
      cross_entropy = tf.reduce_sum(w_ce) #tf.to_float(Q) * tf.reduce_sum(w_ce) / nof 
      
      mask = self._proposal_targets['labels_mask']
      mask = tf.where(mask > 0.1, tf.ones_like(mask), tf.zeros_like(mask)) 
      mask = tf.reshape(mask, (-1,1))

      # RCNN, bbox loss
      pred_bbox = self._predictions['pred_bbox']
      bbox_targets = self._proposal_targets['bbox_targets']
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
      loss_box = self._smooth_l1_loss(pred_bbox, bbox_targets, bbox_inside_weights, bbox_outside_weights)
      
      ## partial scene-graph loss - direct with gt_boxes
      ent_cls_score0 = self._predictions['ent_cls_score0']      
      rel_cls_score0 = self._predictions['rel_cls_score0']
      # entity
      nof_ents = tf.to_float(tf.reduce_sum(self._partial_entity_class)) + 1.0      
      self.nof_ents0 = nof_ents
      ents_for_loss = tf.to_float(tf.reduce_sum(self._partial_entity_class, axis=1))
      self.ents_for_loss0 = ents_for_loss     
      ent_cross_entropy0 = tf.reduce_sum(tf.multiply(ents_for_loss,  tf.nn.softmax_cross_entropy_with_logits(logits=ent_cls_score0, labels=self._partial_entity_class)))# / nof_ents
      self.ent_cross_entropy0 = ent_cross_entropy0
      
      # relation
      partial_rel_class = tf.reshape(self._partial_relation_class, (-1, 43))
      rel_cls_score0 = tf.reshape(rel_cls_score0, (-1, 43))
      rel_ce = tf.nn.softmax_cross_entropy_with_logits(logits=rel_cls_score0, labels=partial_rel_class)

      # positive relation
      nof_rels = tf.to_float(tf.reduce_sum(partial_rel_class[:,:-1])) + 1.0
      self.nof_rels = nof_rels
      rels_for_loss = tf.to_float(tf.reduce_sum(partial_rel_class[:,:-1], axis=1))
      self.rels_for_loss = rels_for_loss
      rel_cross_entropy0 = tf.reduce_sum(tf.multiply(rels_for_loss, rel_ce))# / nof_rels

      # negative relation
      nof_neg_rels = tf.to_float(tf.reduce_sum(partial_rel_class[:,-1])) + 1.0
      factor = 3.0 * tf.to_float(nof_rels) / nof_neg_rels
      neg_rels_for_loss = factor * tf.to_float(partial_rel_class[:,-1])
      rel_cross_entropy0 += tf.reduce_sum(tf.multiply(neg_rels_for_loss, rel_ce))# / nof_neg_rels

      ## partial scene-graph loss - gpi with rpn
      partial_entity_class = self._proposal_targets['partial_entity_class']
      partial_relation_class = self._proposal_targets['partial_relation_class']
      ent_cls_score = self._predictions['ent_cls_score']      
      rel_cls_score = self._predictions['rel_cls_score']
      # entity
      partial_entity_class = tf.multiply(tf.to_float(partial_entity_class), mask)
      self.partial_entity_class = partial_entity_class
      nof_ents = tf.to_float(tf.reduce_sum(partial_entity_class)) + 1.0
      self.nof_ents = nof_ents
      ents_for_loss = tf.to_float(tf.reduce_sum(partial_entity_class, axis=1))
      self.ents_for_loss = ents_for_loss
      
      ent_cross_entropy = tf.reduce_sum(tf.multiply(ents_for_loss,  tf.nn.softmax_cross_entropy_with_logits(logits=ent_cls_score, labels=partial_entity_class)))# / nof_ents
      
      # relation
      N = tf.slice(tf.shape(mask), [0], [1], name="N")
      expand_mask_shape = tf.concat((N, tf.shape(mask)), 0)
      expand_mask = tf.add(tf.zeros(expand_mask_shape), mask)
      expand_mask_transpose = tf.transpose(expand_mask, perm=[1,0,2])
      expand_mask = tf.multiply(expand_mask_transpose, expand_mask)
      self.expand_mask = expand_mask

      partial_relation_class = tf.multiply(tf.to_float(partial_relation_class), expand_mask)
      self.partial_relation_class = partial_relation_class
      partial_rel_class = tf.reshape(partial_relation_class, (-1, 43))
      rel_cls_score = tf.reshape(rel_cls_score, (-1, 43))
      rel_ce = tf.nn.softmax_cross_entropy_with_logits(logits=rel_cls_score, labels=partial_rel_class)

      # positive relation
      nof_rels = tf.to_float(tf.reduce_sum(partial_rel_class[:,:-1])) + 1.0
      self.nof_rels = nof_rels
      rels_for_loss = tf.to_float(tf.reduce_sum(partial_rel_class[:,:-1], axis=1))
      self.rels_for_loss = rels_for_loss
      rel_cross_entropy = tf.reduce_sum(tf.multiply(rels_for_loss, rel_ce))# / nof_rels

      # negative relation
      nof_neg_rels = tf.to_float(tf.reduce_sum(partial_rel_class[:,-1])) + 1.0
      factor = 3.0 * tf.to_float(nof_rels) / nof_neg_rels
      neg_rels_for_loss = factor * tf.to_float(partial_rel_class[:,-1])
      rel_cross_entropy += tf.reduce_sum(tf.multiply(neg_rels_for_loss, rel_ce))# / nof_neg_rels

      ## sum losses
      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box
      self._losses['rel_cross_entropy'] = rel_cross_entropy
      self._losses['ent_cross_entropy'] = ent_cross_entropy
      self._losses['rel_cross_entropy0'] = rel_cross_entropy0
      self._losses['ent_cross_entropy0'] = ent_cross_entropy0
      
      loss = rpn_cross_entropy + rpn_loss_box + loss_box + 0.01 * ent_cross_entropy0 + 0.01 * rel_cross_entropy0 + 0.2 * cross_entropy
      #loss = rpn_cross_entropy + rpn_loss_box  + 0.2 * cross_entropy
      #loss = rpn_cross_entropy + rpn_loss_box + 0.2 * cross_entropy + 0.01 * ent_cross_entropy + 0.01 * rel_cross_entropy + 0.01 * ent_cross_entropy0 + 0.01 * rel_cross_entropy0 # / tf.to_float(tf.shape(self._proposal_targets["labels"])[0])
      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
      self._losses['total_loss'] = loss# + regularization_loss

      self._event_summaries.update(self._losses)

    return loss

  def _region_proposal(self, net_conv, is_training, initializer):
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
    self._act_summaries.append(rpn)
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')
    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    if is_training:
      _rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
      # Try to have a deterministic order for the computing graph, for reproducibility
      with tf.control_dependencies([rpn_labels]):
        rois, _ = self._proposal_target_layer(_rois, roi_scores, "rpn_rois")
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      else:
        raise NotImplementedError

    self._predictions["rpn"] = rpn
    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois
    self._predictions["_rois"] = _rois
    self._predictions["roi_scores"] = roi_scores
    return rois

  def _region_classification(self, orig_fc7, expand_node_features, is_training, initializer, initializer_bbox):
    expand_node_features = slim.fully_connected(expand_node_features, 500,
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       scope='cls_score__new')
    cls_score = slim.fully_connected(expand_node_features, self._num_classes,
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score2__new')
    cls_prob = self._softmax_layer(cls_score, "cls_prob")
    cls_pred = tf.argmax(cls_score, axis=2, name="cls_pred")

    pred_bbox = slim.fully_connected(orig_fc7, 4,
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred__new')
    self._predictions["fc7"] = orig_fc7
    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["pred_bbox"] = pred_bbox


    return cls_prob, pred_bbox

  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None):
    raise NotImplementedError

  def create_architecture(self, mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 4])
    self._gt_labels = tf.placeholder(tf.float32, shape=[None, None, 1])

    self._query = tf.placeholder(tf.float32, shape=[None, 235])
    self._partial_entity_class = tf.placeholder(tf.float32, shape=[None, 96])
    self._partial_relation_class = tf.placeholder(tf.float32, shape=[None, None, 43])
    self._tag = tag
    

    self._num_classes = num_classes
    self._pw_num_classes = num_classes
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag != None

    # handle most of the regularizers here
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)): 
      rois, cls_prob, bbox_pred = self._build_network(training)

    layers_to_output = {'rois': rois}

    for var in tf.trainable_variables():
      self._train_summaries.append(var)

    if testing:
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (1))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (1))
      self._predictions["bbox_pred"] *= stds
      self._predictions["bbox_pred"] += means
    else:
      self._add_losses()
      layers_to_output.update(self._losses)

      val_summaries = []
      with tf.device("/cpu:0"):
        val_summaries.append(self._add_gt_image_summary())
        for key, var in self._event_summaries.items():
          val_summaries.append(tf.summary.scalar(key, var))
        for key, var in self._score_summaries.items():
          self._add_score_summary(key, var)
        for var in self._act_summaries:
          self._add_act_summary(var)
        for var in self._train_summaries:
          self._add_train_summary(var)

      self._summary_op = tf.summary.merge_all()
      self._summary_op_val = tf.summary.merge(val_summaries)

    layers_to_output.update(self._predictions)

    return layers_to_output

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["head"], feed_dict=feed_dict)
    return feat

  # only useful during testing mode
  def test_image(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}

    cls_score, cls_prob, pw_cls_score, pw_cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions["pw_cls_score"],
                                                     self._predictions['pw_cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, pw_cls_score, pw_cls_prob, bbox_pred, rois

  def get_summary(self, sess, blobs):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'], self._gt_labels: blobs['gt_labels'], self._query: blobs['query']}
    summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

    return summary

  def train_step(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'], self._gt_labels: blobs['gt_labels'], self._query: blobs['query'],
                 self._partial_entity_class: blobs['partial_entity_class'], self._partial_relation_class: blobs['partial_relation_class']}
    if train_op != None:
      rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_ent, loss_rel, loss_ent0, loss_rel0, loss, pred_label, pred, pred_prob, pred_boxes0, pred_boxes, ent, rel, ent0, rel0, rois, ent_class, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                          self._losses['rpn_loss_box'],
                                                                          self._losses['cross_entropy'],
                                                                          self._losses['loss_box'],
                                                                          self._losses['ent_cross_entropy'],
                                                                          self._losses['rel_cross_entropy'],
                                                                          self._losses['ent_cross_entropy0'],
                                                                          self._losses['rel_cross_entropy0'],
                                                                          self._losses['total_loss'],
                                                                          self._proposal_targets["labels"],
                                                                          self._predictions["cls_pred"],
                                                                          self._predictions["cls_prob"],
                                                                          self._predictions['pred_bbox0_pool5'],
                                                                          self._predictions['pred_bbox'],
                                                                          self._predictions['ent_cls_score'],
                                                                          self._predictions['rel_cls_score'],
                                                                          self._predictions['ent_cls_score0'],
                                                                          self._predictions['rel_cls_score0'],
                                                                          self._predictions['rois'],
                                                                          self._proposal_targets['partial_entity_class'],
                                                                          train_op], feed_dict=feed_dict)
    else:
      rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_ent, loss_rel, loss_ent0, loss_rel0, loss, pred_label, pred, pred_prob, pred_boxes0, pred_boxes, ent, rel, ent0, rel0, rois, ent_class = sess.run([self._losses["rpn_cross_entropy"],
                                                                          self._losses['rpn_loss_box'],
                                                                          self._losses['cross_entropy'],
                                                                          self._losses['loss_box'],
                                                                          self._losses['ent_cross_entropy'],
                                                                          self._losses['rel_cross_entropy'],
                                                                          self._losses['ent_cross_entropy0'],
                                                                          self._losses['rel_cross_entropy0'],
                                                                          self._losses['total_loss'],
                                                                          self._proposal_targets["labels"],
                                                                          self._predictions["cls_pred"],
                                                                          self._predictions["cls_prob"],
                                                                          self._predictions['pred_bbox0_pool5'],
                                                                          self._predictions['pred_bbox'],
                                                                          self._predictions['ent_cls_score'],
                                                                          self._predictions['rel_cls_score'],
                                                                          self._predictions['ent_cls_score0'],
                                                                          self._predictions['rel_cls_score0'],
                                                                          self._predictions['rois'],
                                                                          self._proposal_targets['partial_entity_class']], feed_dict=feed_dict)

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_ent, loss_rel, loss_ent0, loss_rel0, loss, pred_label, pred, pred_prob, pred_boxes0, pred_boxes, ent, rel, ent0, rel0, rois

  def train_step_with_summary(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'], self._gt_labels: blobs['gt_labels'], self._query: blobs['query']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                 self._losses['rpn_loss_box'],
                                                                                 self._losses['cross_entropy'],
                                                                                 self._losses['loss_box'],
                                                                                 self._losses['total_loss'],
                                                                                 self._summary_op,
                                                                                 train_op],
                                                                                feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'], self._gt_labels: blobs['gt_labels'], self._query: blobs['query']}
    sess.run([train_op], feed_dict=feed_dict)

  def union_box(self, x1, y1, x2, y2):
    sub_x1, obj_x1 = tf.meshgrid(x1, x1)
    sub_y1, obj_y1 = tf.meshgrid(y1, y1)
    sub_x2, obj_x2 = tf.meshgrid(x2, x2)
    sub_y2, obj_y2 = tf.meshgrid(y2, y2)

    pw_x1 = tf.minimum(sub_x1, obj_x1)
    pw_y1 = tf.minimum(sub_y1, obj_y1)
    pw_x2 = tf.maximum(sub_x2, obj_x2)
    pw_y2 = tf.maximum(sub_y2, obj_y2)

    return pw_x1, pw_y1, pw_x2, pw_y2

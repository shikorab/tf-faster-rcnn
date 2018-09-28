# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer

try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import os
import sys
import glob
import time
import math

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from model.bbox_transform import clip_boxes, bbox_transform_inv
from utils.cython_bbox import bbox_overlaps


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    # y1 >= 0
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    # x2 < im_shape[1]
    boxes[:, 2] = np.minimum(boxes[:, 2], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3] = np.minimum(boxes[:, 3], im_shape[0] - 1)
    return boxes


class SolverWrapper(object):
    """
    A wrapper class for the training process
  """

    def __init__(self, sess, network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.valroidb = valroidb
        self.output_dir = output_dir
        self.tbdir = tbdir
        # Simply put '_val' at the end to save the summaries from the validation set
        self.tbvaldir = tbdir + '_val'
        if not os.path.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)
        self.pretrained_model = pretrained_model

    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # current position in the database
        cur = self.data_layer._cur
        # current shuffled indexes of the database
        perm = self.data_layer._perm
        # current position in the validation database
        cur_val = self.data_layer_val._cur
        # current shuffled indexes of the validation database
        perm_val = self.data_layer_val._perm

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def from_snapshot(self, sess, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        init = tf.global_variables_initializer()
        sess.run(init)
        self.saver.restore(sess, sfile)
        print('Restored.')
        # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
        # tried my best to find the random states so that it can be recovered exactly
        # However the Tensorflow state is currently not available
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            cur = pickle.load(fid)
            perm = pickle.load(fid)
            cur_val = pickle.load(fid)
            perm_val = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)

            np.random.set_state(st0)
            self.data_layer._cur = cur
            self.data_layer._perm = perm
            self.data_layer_val._cur = cur_val
            self.data_layer_val._perm = perm_val

        return last_snapshot_iter

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)
            # Build the main computation graph
            layers = self.net.create_architecture('TRAIN', self.imdb.num_classes, tag='default',
                                                  anchor_scales=cfg.ANCHOR_SCALES,
                                                  anchor_ratios=cfg.ANCHOR_RATIOS)
            # Define the loss
            loss = layers['total_loss']
            # Set learning rate and momentum
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

            # Compute the gradients with regard to the loss
            gvs = self.optimizer.compute_gradients(loss)
            # Double the gradient of the bias if set
            if cfg.TRAIN.DOUBLE_BIAS:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult') as scope:
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = self.optimizer.apply_gradients(final_gvs)
            else:
                train_op = self.optimizer.apply_gradients(gvs)

            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tbvaldir)

        return lr, train_op

    def find_previous(self, max_iters):
        sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_LOAD_PREFIX + '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for stepsize in range(0, max_iters, cfg.TRAIN.SNAPSHOT_ITERS):
            redfiles.append(os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_LOAD_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss in redfiles]

        nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_LOAD_PREFIX + '_iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

    def initialize(self, sess):
        # Initial file lists are empty
        np_paths = []
        ss_paths = []
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(self.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        # Get the variables to restore, ignoring the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
        init = tf.global_variables_initializer()
        sess.run(init)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, self.pretrained_model)
        print('Loaded.')
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights
        self.net.fix_variables(sess, self.pretrained_model)
        print('Fixed.')
        last_snapshot_iter = 0
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = list(cfg.TRAIN.STEPSIZE)

        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def restore(self, sess, sfile, nfile):
        # Get the most recent snapshot and restore
        np_paths = [nfile]
        ss_paths = [sfile]
        # Restore model from snapshots
        last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
        self.net.fix_variables(sess, sfile)
        print('Fixed.')
        # Set the learning rate
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = cfg.TRAIN.STEPSIZE
        #for stepsize in cfg.TRAIN.STEPSIZE:
        #    if last_snapshot_iter > stepsize:
        #        rate *= cfg.TRAIN.GAMMA
        #    else:
        #       stepsizes.append(stepsize)

        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

        to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)

    def train_model(self, sess, max_iters):
        # Build data layers for both training and validation set
        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
        self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

        # Construct the computation graph
        lr, train_op = self.construct_graph(sess)

        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous(max_iters)

        # Initialize the variables or restore them from the last snapshot
        if lsf == 0:
            variables = tf.global_variables()
            var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
            variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
            
            self.saver = tf.train.Saver(variables_to_restore, max_to_keep=100000, reshape=True)
            
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
        else:
            variables = tf.global_variables()
            var_keep_dic = self.get_variables_in_checkpoint_file(sfiles[-1])
            variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
            #variables_to_restore = var_keep_dic
            self.saver = tf.train.Saver(variables_to_restore, max_to_keep=100000, reshape=True)
            
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess,
                                                                                   str(sfiles[-1]),
                                                                                  str(nfiles[-1]))
        self.saver = tf.train.Saver(max_to_keep=100000, reshape=True) 
        iter = 1
        # Make sure the lists are not empty
        stepsizes.append(max_iters)
        stepsizes.reverse()
        next_stepsize = stepsizes.pop()

        while iter < max_iters + 1:
            self.run_epoch(iter, self.data_layer, "train", sess, lr, train_op)
            self.run_epoch(iter, self.data_layer_val, "validation", sess, lr, None)

            if iter == next_stepsize + 1:
                # Add snapshot here before reducing the learning rate
                self.snapshot(sess, iter)
                rate *= cfg.TRAIN.GAMMA
                sess.run(tf.assign(lr, rate))
                next_stepsize = stepsizes.pop()

            # Snapshotting
            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self.snapshot(sess, iter)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                try:
                    if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                        self.remove_snapshot(np_paths, ss_paths)
                except:
                    print("failed to remove snapshot")

            iter += 1

        if last_snapshot_iter != iter - 1:
            self.snapshot(sess, iter - 1)

        self.writer.close()
        self.valwriter.close()

    def run_epoch(self, epoch, data_layer, name, sess, lr, train_op):
        epoch_total_loss = 0.0
        epoch_rpn_loss_cls = 0.0
        epoch_rpn_loss_box = 0.0
        epoch_loss_cls = 0.0
        epoch_loss_box = 0.0
        epoch_iter = 0.0
        epoch_rpn_overlaps = 0.0
        epoch_rpn_overlaps0 = 0.0
        epoch_ent = 0.0
        epoch_rel = 0.0
        epoch_ent_accuracy = 0.0
        epoch_rel_accuracy = 0.0
        epoch_ent0 = 0.0
        epoch_rel0 = 0.0
        epoch_ent0_accuracy = 0.0
        epoch_rel0_accuracy = 0.0
        accum_results = None

        timer = Timer()
        # Get training data, one batch at a time
        while True:
            timer.tic()

            blobs, new_epoch = data_layer.forward()
            
            if new_epoch:
                sub_iou = float(accum_results['sub_iou']) / accum_results['total']
                obj_iou = float(accum_results['obj_iou']) / accum_results['total']
                acc = float(accum_results['acc']) / accum_results['total']
                acc0 = float(accum_results['acc0']) / (accum_results['acc0_total'] + 1.0)
                acc1 = float(accum_results['acc1']) / (accum_results['acc1_total'] + 1.0)
                acc2 = float(accum_results['acc2']) / (accum_results['acc2_total'] + 1.0)
                acc3 = float(accum_results['acc3']) / (accum_results['acc3_total'] + 1.0)


                print('%s (%s): epoch %d iter: %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                      '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> loss_ent: %.6f loss_rel: %.6f ent_acc: %.6f rel_acc: %.6f\n >>> loss_ent0: %.6f loss_rel0: %.6f ent0_acc: %.6f rel0_acc: %.6f \n >>> lr: %f' % \
                      (name, cfg.TRAIN.SNAPSHOT_PREFIX, epoch, int(epoch_iter), epoch_total_loss / epoch_iter, epoch_rpn_loss_cls / epoch_iter,
                       epoch_rpn_loss_box / epoch_iter, epoch_loss_cls / epoch_iter, epoch_loss_box / epoch_iter, epoch_ent / epoch_iter, epoch_rel / epoch_iter, epoch_ent_accuracy / epoch_iter, epoch_rel_accuracy / epoch_iter, epoch_ent0 / epoch_iter, epoch_rel0 / epoch_iter, epoch_ent0_accuracy / epoch_iter, epoch_rel0_accuracy / epoch_iter,lr.eval()))
                print('sub_iou: {} obj_iou {} rpn_overlaps {} rpn_overlaps0 {}'.format(sub_iou, obj_iou, epoch_rpn_overlaps / epoch_iter, epoch_rpn_overlaps0 / epoch_iter))
                print('acc: {} acc0: {} acc1: {} acc2: {} acc3: {}'.format(acc, acc0, acc1, acc2, acc3))
                return
	    
            if blobs["query"].shape[0] == 0 or blobs["gt_boxes"].shape[0] == 0:
               continue 

            # Compute the graph without summary
            try:
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_ent, loss_rel, loss_ent0, loss_rel0, total_loss, pred_label, pred, pred_prob, pred_boxes0, pred_boxes, ent, rel, ent0, rel0, rois = self.net.train_step(sess, blobs, train_op)
                if math.isnan(total_loss):
                    print("total loss is nan - iter %d" % (int(epoch_iter)))
                    continue

                gt_bbox = blobs['gt_boxes']
                gt = blobs['gt_labels'][:, :, 0]
                im = blobs["im_info"]
                gt_ent = blobs['partial_entity_class']
                gt_rel = blobs['partial_relation_class']
                boxes = rois[:, 1:5]
                pred_boxes = np.reshape(pred_boxes, [pred_boxes.shape[0], -1])
                # Apply bounding-box regression deltas
                box_deltas = pred_boxes * np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS) + np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                pred_boxes = bbox_transform_inv(boxes, box_deltas)
                pred_boxes = _clip_boxes(pred_boxes, im)
                pred_boxes[:, 0] /= im[1]
                pred_boxes[:, 1] /= im[0]
                pred_boxes[:, 2] /= im[1]
                pred_boxes[:, 3] /= im[0]
                gt_bbox_norm = gt_bbox.copy()
                gt_bbox_norm[:, 0] /= im[1]
                gt_bbox_norm[:, 1] /= im[0]
                gt_bbox_norm[:, 2] /= im[1]
                gt_bbox_norm[:, 3] /= im[0]
                rpn_overlaps0, rpn_overlaps, ent_accuracy, rel_accuracy, ent0_accuracy, rel0_accuracy = rpn_test(gt_bbox_norm, gt_ent, pred_boxes0, pred_boxes, ent, ent0, gt_rel, rel, rel0)
                for query_index in range(gt.shape[0]):
                    results = iou_test(gt[query_index], gt_bbox_norm, pred_label[query_index, :, 0], pred[query_index], pred_prob[query_index], pred_boxes0, blobs['im_info'])
                    # accumulate results
                    if accum_results is None:
                        accum_results = results
                    else:
                        for key in results:
                            accum_results[key] += results[key]
                epoch_iter += 1.0
                epoch_loss_box += loss_box
                epoch_rpn_loss_box += rpn_loss_box
                epoch_loss_cls += loss_cls
                epoch_rpn_loss_cls += rpn_loss_cls
                epoch_total_loss += total_loss
                epoch_rpn_overlaps += rpn_overlaps
                epoch_rpn_overlaps0 += rpn_overlaps0 
                epoch_ent_accuracy += ent_accuracy
                epoch_rel_accuracy += rel_accuracy
                epoch_ent0_accuracy += ent0_accuracy
                epoch_rel0_accuracy += rel0_accuracy
                epoch_ent += loss_ent
                epoch_rel += loss_rel
                epoch_ent0 += loss_ent0
                epoch_rel0 += loss_rel0


            except Exception as e:
                print(e)
                print("error iter %d" % (int(epoch_iter)))
                continue

            timer.toc()

            # Display training information
            if epoch == 1 and int(epoch_iter) % (cfg.TRAIN.DISPLAY) == 0:
                sub_iou = float(accum_results['sub_iou']) / accum_results['total']
                obj_iou = float(accum_results['obj_iou']) / accum_results['total']
                acc = float(accum_results['acc']) / accum_results['total']
                acc0 = float(accum_results['acc0']) / (accum_results['acc0_total'] + 1.0)
                acc1 = float(accum_results['acc1']) / (accum_results['acc1_total'] + 1.0)
                acc2 = float(accum_results['acc2']) / (accum_results['acc2_total'] + 1.0)
                acc3 = float(accum_results['acc3']) / (accum_results['acc3_total'] + 1.0)


                print('%s (%s): epoch %d iter: %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                      '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> loss_ent: %.6f loss_rel: %.6f ent_acc: %.6f rel_acc: %.6f\n >>> loss_ent0: %.6f loss_rel0: %.6f ent0_acc: %.6f rel0_acc: %.6f \n >>> lr: %f' % \
                      (name, cfg.TRAIN.SNAPSHOT_PREFIX, epoch, int(epoch_iter), epoch_total_loss / epoch_iter, epoch_rpn_loss_cls / epoch_iter,
                       epoch_rpn_loss_box / epoch_iter, epoch_loss_cls / epoch_iter, epoch_loss_box / epoch_iter, epoch_ent / epoch_iter, epoch_rel / epoch_iter, epoch_ent_accuracy / epoch_iter, epoch_rel_accuracy / epoch_iter, epoch_ent0 / epoch_iter, epoch_rel0 / epoch_iter, epoch_ent0_accuracy / epoch_iter, epoch_rel0_accuracy / epoch_iter,lr.eval()))
                print('sub_iou: {} obj_iou {} rpn_overlaps {} rpn_overlaps0 {}'.format(sub_iou, obj_iou, epoch_rpn_overlaps / epoch_iter, epoch_rpn_overlaps0 / epoch_iter))
                print('acc: {} acc0: {} acc1: {} acc2: {} acc3: {}'.format(acc, acc0, acc1, acc2, acc3))

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after))
    return roidb


def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
    """Train a Faster R-CNN network."""
    roidb = filter_roidb(roidb)
    valroidb = filter_roidb(valroidb)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir, tb_dir,
                           pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(sess, max_iters)
        print('done solving')


MASK_WIDTH = 32
def softmax(x):
    xexp = np.exp(x)
    return xexp / np.sum(xexp, axis=-1, keepdims=1)
    

def rpn_test(gt_bbox, gt_ent, pred_boxes0, pred_boxes, pred_ent, pred_ent0, gt_rel, pred_rel, pred_rel0):
    overlaps0 = bbox_overlaps(np.ascontiguousarray(gt_bbox, dtype=np.float), np.ascontiguousarray(pred_boxes0, dtype=np.float))
    overlaps0_assign = np.argmax(overlaps0, axis=1)
    max_overlaps0 = overlaps0.max(axis=1)

    overlaps = bbox_overlaps(np.ascontiguousarray(gt_bbox, dtype=np.float), np.ascontiguousarray(pred_boxes, dtype=np.float))
    overlaps_assign = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps.max(axis=1)

    pred_ent = softmax(pred_ent[overlaps_assign])
    ent_accuracy = np.sum(np.multiply(pred_ent, gt_ent)) / np.sum(gt_ent)
    
    pred_rel = softmax(pred_rel[overlaps_assign,:][:,overlaps_assign])
    rel_accuracy = np.sum(np.multiply(pred_rel[:,:,:-1], gt_rel[:,:,:-1])) / np.sum(gt_rel[:,:,:-1])
    
    pred_ent0 = softmax(pred_ent0)
    ent0_accuracy = np.sum(np.multiply(pred_ent0, gt_ent)) / np.sum(gt_ent)
    
    pred_rel0 = softmax(pred_rel0)
    rel0_accuracy = np.sum(np.multiply(pred_rel0[:,:,:-1], gt_rel[:,:,:-1])) / np.sum(gt_rel[:,:,:-1])
    
    return np.mean(max_overlaps0), np.mean(max_overlaps), ent_accuracy, rel_accuracy, ent0_accuracy, rel0_accuracy

def iou_test(gt, gt_bbox, pred_label, pred, pred_prob, pred_bbox, im_info):
    results = {}
    # number of objects
    results["total"] = 1

    results["sub_iou"] = 0.0
    results["obj_iou"] = 0.0
    results["acc"] = np.sum(pred == pred_label).astype(float) / pred_label.shape[0] 
    for i in range(4):
        total = np.sum(pred_label == i).astype(float)
        if total != 0:
            results["acc" + str(i)] = np.sum(np.logical_and(pred == pred_label, pred_label == i)).astype(float) / total
            results["acc" + str(i) + "_total"] = 1.0
        else:
            results["acc" + str(i)] = 0.0
            results["acc" + str(i) + "_total"] = 0.0

 
    width = MASK_WIDTH
    height = int(np.ceil(float(MASK_WIDTH) / im_info[1] * im_info[0]))
    MASK_SHAPE = (width, height) 
    mask_sub_gt = np.zeros(MASK_SHAPE, dtype=bool)
    mask_obj_gt = np.zeros(MASK_SHAPE, dtype=bool)
    mask_sub_pred = np.zeros(MASK_SHAPE, dtype=bool)
    mask_obj_pred = np.zeros(MASK_SHAPE, dtype=bool)

    # most probable sub and obj
    i = np.argmax(pred_prob[:, 1])
    mask_sub_pred[int(pred_bbox[i][0] * MASK_SHAPE[0]):int(math.ceil(pred_bbox[i][2] * MASK_SHAPE[0])),
    int(pred_bbox[i][1] * MASK_SHAPE[1]):int(math.ceil(pred_bbox[i][3] * MASK_SHAPE[1]))] = True
    i = np.argmax(pred_prob[:, 2])
    mask_obj_pred[int(pred_bbox[i][0] * MASK_SHAPE[0]):int(math.ceil(pred_bbox[i][2] * MASK_SHAPE[0])),
    int(pred_bbox[i][1] * MASK_SHAPE[1]):int(math.ceil(pred_bbox[i][3] * MASK_SHAPE[1]))] = True

    # GT mask
    for i in range(gt.shape[0]):
        if gt[i] == 1:
            mask_sub_gt[
            int(gt_bbox[i][0] * MASK_SHAPE[0]):int(math.ceil(gt_bbox[i][2] * MASK_SHAPE[0])),
            int(gt_bbox[i][1] * MASK_SHAPE[1]):int(
                math.ceil(gt_bbox[i][3] * MASK_SHAPE[1]))] = True
        if gt[i] == 2:
            mask_obj_gt[
            int(gt_bbox[i][0] * MASK_SHAPE[0]):int(math.ceil(gt_bbox[i][2] * MASK_SHAPE[0])),
            int(gt_bbox[i][1] * MASK_SHAPE[1]):int(
                math.ceil(gt_bbox[i][3] * MASK_SHAPE[1]))] = True

    # predicted mask
    for i in range(pred.shape[0]):
        if pred[i] == 1:
            mask_sub_pred[int(pred_bbox[i][0] * MASK_SHAPE[0]):int(math.ceil(pred_bbox[i][2] * MASK_SHAPE[0])),
            int(pred_bbox[i][1] * MASK_SHAPE[1]):int(math.ceil(pred_bbox[i][3] * MASK_SHAPE[1]))] = True
        if pred[i] == 2:
            mask_obj_pred[int(pred_bbox[i][0] * MASK_SHAPE[0]):int(math.ceil(pred_bbox[i][2] * MASK_SHAPE[0])),
            int(pred_bbox[i][1] * MASK_SHAPE[1]):int(math.ceil(pred_bbox[i][3] * MASK_SHAPE[1]))] = True

    sub_iou = iou(mask_sub_gt, mask_sub_pred)
    obj_iou = iou(mask_obj_gt, mask_obj_pred)

    results["sub_iou"] += sub_iou
    results["obj_iou"] += obj_iou

    return results


def iou(mask_a, mask_b):
    union = np.sum(np.logical_or(mask_a, mask_b))
    if union == 0:
        return 0.0
    intersection = np.sum(np.logical_and(mask_a, mask_b))
    return float(intersection) / float(union)

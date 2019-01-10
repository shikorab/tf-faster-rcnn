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
import keras as k

try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
import sys
import glob
import time
import math
import cv2

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

    def __init__(self, sess, network, imdb, roidb, valroidb, output_dir = "", tbdir = "", pretrained_model=""):
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
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}' % (iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}' % (filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}' % (iter) + '.pkl'
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
        print('Restoring model snapshots from {:s}' % (sfile))
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

    def train_model(self, sess, max_iters, just_test = False):
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
            self.saver = tf.train.Saver(variables_to_restore, max_to_keep=100000, reshape=True)
            
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess,
                                                                                   str(sfiles[-1]),
                                                                                  str(nfiles[-1]))
        self.saver = tf.train.Saver(max_to_keep=100000, reshape=True) 
        iter = 1
        np_paths = [] 
        ss_paths = []
        # Make sure the lists are not empty
        stepsizes.append(max_iters)
        stepsizes.reverse()
        next_stepsize = stepsizes.pop()
        best_result = 0.0

        sess.run(tf.assign(lr, rate))

        while iter < max_iters + 1:
            
            if iter == next_stepsize + 1:
                # Add snapshot here before reducing the learning rate
                rate *= cfg.TRAIN.GAMMA
                sess.run(tf.assign(lr, rate))
                next_stepsize = stepsizes.pop()
            
            if not just_test:
                self.run_epoch(iter, self.data_layer, "train", sess, lr, train_op)
                result = self.run_epoch(iter, self.data_layer_val, "validation", sess, lr, None)
            else:
                self.run_epoch(iter, self.data_layer_val, "test", sess, lr, None) 
                result = - 1.0

            # Snapshotting
            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0 and not just_test:
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
            
            if result > best_result:
                self.snapshot(sess, 0)
                best_result = result
            print(">>> best_result %f" % (best_result))

            iter += 1

        if last_snapshot_iter != iter - 1 and not just_test:
            self.snapshot(sess, iter - 1)

        self.writer.close()
        self.valwriter.close()

    def run_epoch(self, epoch, data_layer, name, sess, lr, train_op):
        accum_results = None
        accum_losses = None
        epoch_iter = 0.0

        image_stat = '2376309.jpg'
        objs_cat = ["giraffe", "bowl", "food", "face", "people", "shirt", "bench", "light", "head", "zebra", "cow", "sign", "motorcycle", "floor", "hat", "sheep", "truck", "water", "chair", "field", "door", "pizza", "tree", "car", "leg", "bag", "fence", "sidewalk", "girl", "leaves", "jacket", "windows", "road", "glass", "bed", "sand", "trees", "player", "helmet", "man", "grass", "cake", "bear", "hand", "cloud", "street", "ground", "airplane", "mirror", "clock", "plate", "ear", "hair", "window", "boy", "clouds", "handle", "counter", "glasses", "pants", "eye", "pole", "line", "wall", "animal", "shadow", "train", "bike", "boat", "horse", "tail", "nose", "beach", "snow", "elephant", "bottle", "surfboard", "cat", "skateboard", "shorts", "woman", "bird", "sky", "shelf", "tracks", "kite", "umbrella", "guy", "building", "dog", "background", "table", "child", "lady", "plane", "desk", "bus", "wheel", "arm", "person"]

        rels_cat = ["wearing a", "made of", "on front of", "with a", "WEARING", "above", "carrying", "has an", "covering", "and", "wears", "around", "with", "laying on", "inside", "attached to", "at", "on a", "of a", "hanging on", "near", "OF", "sitting on", "of", "next to", "riding", "under", "over", "behind", "sitting in", "ON", "eating", "to", "in a", "has", "parked on", "covered in", "holding", "for", "playing", "against", "by", "from", "has a", "standing on", "on side of", "in", "wearing", "watching", "walking on", "beside", "below", "IN", "mounted on", "have", "are on", "are in", "in front of", "looking at", "belonging to", "on top of", "holds", "inside of", "along", "hanging from", "standing in", "says", "painted on", "between", "on"]
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255), (0, 0, 0), (255, 255, 255), (128, 128, 128), (128, 0, 0), (0, 128, 0), (0, 0, 128)]
        timer = Timer()
        # Get training data, one batch at a time
        while True:
            timer.tic()

            blobs, new_epoch = data_layer.forward()
            if blobs["orig_image"].split(".")[0] not in ["150500"]:#, "2396405", "2384930", "2396809", "2332739", "2320584", "2408814", "2350169", "2340600"]:
                continue
            
            if new_epoch and epoch_iter != 0.0:
                print_stat(name, epoch, epoch_iter, lr, accum_results, accum_losses)
                sub_iou = float(accum_results['sub_iou']) / accum_results['total']
                obj_iou = float(accum_results['obj_iou']) / accum_results['total']
                return (sub_iou + obj_iou) / 2
            
            if blobs["query"].shape[0] == 0 or blobs["gt_boxes"].shape[0] == 0:
               continue 

            # Compute the graph without summary
            try:
                losses, predictions, proposal_targets, gpi_attention = self.net.train_step(sess, blobs, train_op)
                if math.isnan(losses["total_loss"]):
                    print("total loss is nan - iter %d" % (int(epoch_iter)))
                    continue

                gt_bbox = blobs['gt_boxes']
                gt = blobs['gt_labels'][:, :, 0]
                im = blobs["im_info"]
                gt_ent = blobs['partial_entity_class']
                gt_rel = blobs['partial_relation_class']
                boxes0 = predictions["rois"][:, 1:5]
                
                
                # Apply bounding-box regression deltas
                pred_boxes = np.reshape(predictions["pred_bbox_gpi"], [predictions["pred_bbox_gpi"].shape[0], -1])
                
                box_deltas = pred_boxes * np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS) + np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                pred_boxes = bbox_transform_inv(boxes0, box_deltas)
                pred_boxes = _clip_boxes(pred_boxes, im)

                # normalize boxes to [0-1]
                gt_bbox_norm = gt_bbox.copy()
                boxes0_norm = boxes0.copy()
                pred_boxes_norm = pred_boxes.copy()
                for i in range(4):
                    pred_boxes_norm[:, i] /= im[(i+1)%2]
                    gt_bbox_norm[:, i] /= im[(i+1)%2]
                    boxes0_norm[:, i] /= im[(i+1)%2]
    
                                    
                rpn_results = rpn_test(gt_bbox_norm, boxes0_norm, pred_boxes_norm, gt_ent, gt_rel, predictions)
                if accum_results == None:
                    first_result = True
                    accum_results = rpn_results
                    accum_losses = losses
                else:
                    first_result = False
                    for key in rpn_results:
                        accum_results[key] += rpn_results[key]
                    for key in losses:
                        accum_losses[key] += losses[key]
                        
                first_img_result = True
                img_results = {}
                for query_index in range(gt.shape[0]):
                    results = iou_test(gt[query_index], gt_bbox_norm,
                                       proposal_targets["labels_mask"], 
                                       proposal_targets["labels"][query_index, :, 0], 
                                       predictions["cls_pred_gpi"][query_index], 
                                       predictions["cls_prob_gpi"][query_index], 
                                       pred_boxes_norm, blobs['im_info'])
                    results_baseline = iou_test(gt[query_index], gt_bbox_norm,
                                       proposal_targets["labels_mask"], 
                                       proposal_targets["labels"][query_index, :, 0], 
                                       predictions["cls_pred_baseline"][query_index], 
                                       predictions["cls_prob_baseline"][query_index], 
                                       boxes0_norm, blobs['im_info'])
                    # accumulate results
                    if first_result:
                        for key in results:
                            accum_results[key] = results[key]
                        for key in results_baseline:
                            accum_results[key + "_bl"] = results_baseline[key]
                        first_result = False

                    else:
                        for key in results:
                            accum_results[key] += results[key]
                        for key in results_baseline:
                            accum_results[key + "_bl"] += results_baseline[key]

                    # accumulate results
                    if first_img_result:
                        for key in results:
                            img_results[key] = results[key]
                        for key in results_baseline:
                            img_results[key + "_bl"] = results_baseline[key]
                        first_img_result = False

                    else:
                        for key in results:
                            img_results[key] += results[key]
                        for key in results_baseline:
                            img_results[key + "_bl"] += results_baseline[key]
                    
                acc = float(img_results['acc']) / img_results['total']
                nof_queries = blobs['query'].shape[0]
                sub_iou = float(img_results['sub_iou']) / img_results['total']
                obj_iou = float(img_results['obj_iou']) / img_results['total']
                if True: #sub_iou + obj_iou < 0.3: #(sub_iou <= 0.85 or obj_iou <= 0.85) and sub_iou > 0.75 and obj_iou > 0.75:
                    path = blobs["orig_image"].split(".")[0]
                    os.mkdir(path)
                    file_str = os.path.join(path, "stat.txt")
                    with open(file_str, 'a') as f:
                        cv2.imwrite(os.path.join(path, "img_orig.jpg"), blobs["data"][0].copy() + cfg.PIXEL_MEANS)
                        f.write("image = %s\n" % (blobs["orig_image"]))
                        f.write("queries (%d)\n" % (nof_queries))
                        for q in range(nof_queries):
                            sub = np.argmax(blobs['query'][q, :100])
                            rel = np.argmax(blobs['query'][q, 100:170])
                            obj = np.argmax(blobs['query'][q, 170:270])
                            f.write("query %d: %d-%d-%d %s-%s-%s\n" % (q, sub, rel, obj, objs_cat[sub], rels_cat[rel], objs_cat[obj]))

                        f.write("gt boxes\n")
                        f.write("------\n")
                        f.write(str(gt_bbox) + "\n")
                        img_gt = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                        for bb in range(gt_bbox.shape[0]):
                            cv2.rectangle(img_gt, (gt_bbox[bb, 0], gt_bbox[bb, 1]), (gt_bbox[bb, 2], gt_bbox[bb, 3]), color[bb % len(color)], 5)
                        cv2.imwrite(os.path.join(path, "img_gt.jpg"), img_gt)
                         
                        f.write("boxes0\n")
                        f.write("------\n")
                        f.write(str(boxes0) + "\n")
                        img_boxes0 = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                        for bb in range(boxes0.shape[0]):
                            if predictions["cls_pred_baseline"][0][bb] != 0:
                                cv2.rectangle(img_boxes0, (int(boxes0[bb, 0]), int(boxes0[bb, 1])), (int(boxes0[bb, 2]), int(boxes0[bb, 3])), color[bb % len(color) ], 5)
                        cv2.imwrite(os.path.join(path, "img_boxes0.jpg"), img_boxes0)
                         
                        f.write("pred_boxes\n")
                        f.write("------\n")
                        f.write(str(pred_boxes) + "\n")
                        img_pred = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                        for bb in range(pred_boxes.shape[0]):
                            if predictions["cls_pred_gpi"][0][bb] != 0:
                                    cv2.rectangle(img_pred, (int(pred_boxes[bb, 0]), int(pred_boxes[bb, 1])), (int(pred_boxes[bb, 2]), int(pred_boxes[bb, 3])), color[bb % len(color)], 5)
                        cv2.imwrite(os.path.join(path, "img_pred_boxes.jpg"), img_pred)
                        
                        img_pred = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                        for bb in range(pred_boxes.shape[0]):
                            if predictions["cls_pred_gpi"][0][bb] == 0:
                                    cv2.rectangle(img_pred, (int(pred_boxes[bb, 0]), int(pred_boxes[bb, 1])), (int(pred_boxes[bb, 2]), int(pred_boxes[bb, 3])), color[bb % len(color)], 5)
                        cv2.imwrite(os.path.join(path, "img_pred_filtered_boxes.jpg"), img_pred)
                        
                        f.write("gt entity labels\n")
                        f.write("------\n")
                        f.write(str(np.argmax(gt_ent, axis=1)) + "\n")
                        f.write("gt realtion labels\n")
                        f.write("------\n")
                        f.write(str(np.argmax(gt_rel, axis=2)) + "\n")
                        f.write("gt rr labels\n")
                        f.write("------\n")
                        f.write(str(proposal_targets["labels"][:, :, 0]) + "\n")
                        f.write("rr prediction\n")
                        f.write("------\n")
                        subsq = [3, 4]
                        objsq = [0, 24] 
                        for q_i in range(nof_queries):
                            sub = objs_cat[np.argmax(blobs['query'][q_i, :100])]
                            rel = rels_cat[np.argmax(blobs['query'][q_i, 100:170])]
                            obj = objs_cat[np.argmax(blobs['query'][q_i, 170:270])]
                            #prediction
                            if np.where(predictions["cls_pred_gpi"][q_i] == 1)[0].shape[0] == 0:
                                subs = np.asarray([np.argmax(predictions["cls_prob_gpi"][q_i, :, 1])])  
	                    else:
                                subs = np.where(predictions["cls_pred_gpi"][q_i] == 1)[0]
                            if np.where(predictions["cls_pred_gpi"][q_i] == 2)[0].shape[0] == 0:
                                objs = np.asarray([np.argmax(predictions["cls_prob_gpi"][q_i, :, 2])])
	                    else:
                                objs = np.where(predictions["cls_pred_gpi"][q_i] == 2)[0]
                            f.write("query %d - sub=%s\n" % (q_i, subs))
                            f.write("query %d - objs=%s\n" % (q_i, objs))
                            img_q = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                            
                            for bb in range(subs.shape[0]):
                                cv2.rectangle(img_q, (int(pred_boxes[subs[bb], 0]), int(pred_boxes[subs[bb], 1])), (int(pred_boxes[subs[bb], 2]), int(pred_boxes[subs[bb], 3])), (255, 0, 0), 5)
                            for bb in range(objs.shape[0]):
                                cv2.rectangle(img_q, (int(pred_boxes[objs[bb], 0]), int(pred_boxes[objs[bb], 1])), (int(pred_boxes[objs[bb], 2]), int(pred_boxes[objs[bb], 3])), (0, 255, 0), 5)
                            cv2.imwrite(os.path.join(path, "pred_query_%d-%s-%s-%s.jpg" % (q_i, sub, rel, obj)), img_q)
                            #gt
                            subs  = np.where(blobs['gt_labels'][q_i] == 1)[0]
                            objs  = np.where(blobs['gt_labels'][q_i] == 2)[0]
                            img_q_gt = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                            for bb in range(subs.shape[0]):
                                cv2.rectangle(img_q_gt, (int(gt_bbox[subs[bb], 0]), int(gt_bbox[subs[bb], 1])), (int(gt_bbox[subs[bb], 2]), int(gt_bbox[subs[bb], 3])), (255, 0, 0), 5)
                            for bb in range(objs.shape[0]):
                                cv2.rectangle(img_q_gt, (int(gt_bbox[objs[bb], 0]), int(gt_bbox[objs[bb], 1])), (int(gt_bbox[objs[bb], 2]), int(gt_bbox[objs[bb], 3])), (0, 255, 0), 5)
                            cv2.imwrite(os.path.join(path, "gt_query_%d-%s-%s-%s.jpg" % (q_i, sub, rel, obj)), img_q_gt)
                            #neig attention
                            img_neig_atten_sub = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                            border = np.argsort(gpi_attention["neighbour"])[subsq[q_i], ::-1]
                            for bb_i in range(min(5, pred_boxes.shape[0])):
                                bb = border[bb_i]
                                cv2.rectangle(img_neig_atten_sub, (int(pred_boxes[bb, 0]), int(pred_boxes[bb, 1])), (int(pred_boxes[bb, 2]), int(pred_boxes[bb, 3])), color[bb % len(color) ], 5)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(img_neig_atten_sub,str(bb_i),(int(pred_boxes[bb, 0]+5), int(pred_boxes[bb, 1]+25)), font, 1,(255,255,255),2,cv2.LINE_AA)
                            cv2.imwrite(os.path.join(path, "sub_neig_atten_query_%d-%s-%s-%s.jpg" % (q_i, sub, rel, obj)), img_neig_atten_sub)
                            
                            img_neig_atten_obj = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                            border = np.argsort(gpi_attention["neighbour"])[objs[0], ::-1]
                            for bb_i in range(min(4, pred_boxes.shape[0])):
                                bb = border[bb_i]
                                cv2.rectangle(img_neig_atten_obj, (int(pred_boxes[bb, 0]), int(pred_boxes[bb, 1])), (int(pred_boxes[bb, 2]), int(pred_boxes[bb, 3])), color[bb % len(color) ], 5)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(img_neig_atten_obj,str(bb_i),(int(pred_boxes[bb, 0]+5), int(pred_boxes[bb, 1]+25)), font, 1,(255,255,255),2,cv2.LINE_AA)
                            cv2.imwrite(os.path.join(path, "obj_neig_atten_query_%d-%s-%s-%s.jpg" % (q_i, sub, rel, obj)), img_neig_atten_obj)
                            
                        f.write(str(predictions["cls_pred_gpi"]) + "\n")
                        sub_iou = float(img_results['sub_iou']) / img_results['total']
                        obj_iou = float(img_results['obj_iou']) / img_results['total']
                        prec0 = 1.0
                        prec1 = 1.0
                        prec2 = 1.0
                        prec3 = 1.0
                        recall0 = 1.0
                        recall1 = 1.0
                        recall2 = 1.0
                        recall3 = 1.0
                        if img_results['prec0_total'] != 0:
                            prec0 = float(img_results['prec0']) / (img_results['prec0_total'])
                        if img_results['prec1_total'] != 0:
                            prec1 = float(img_results['prec1']) / (img_results['prec1_total'])
                        if img_results['prec2_total'] != 0:
                            prec2 = float(img_results['prec2']) / (img_results['prec2_total'])
                        if img_results['prec3_total'] != 0:
                            prec3 = float(img_results['prec3']) / (img_results['prec3_total'])
                        if img_results['recall0_total'] != 0:
                            recall0 = float(img_results['recall0']) / (img_results['recall0_total'])
                        if img_results['recall1_total'] != 0:
                            recall1 = float(img_results['recall1']) / (img_results['recall1_total'])
                        if img_results['recall2_total'] != 0:
                            recall2 = float(img_results['recall2']) / (img_results['recall2_total'])
                        if img_results['recall3_total'] != 0:
                            recall3 = float(img_results['recall3']) / (img_results['recall3_total'])
                        f.write('sub_iou: %.4f obj_iou: %.4f rpn_overlaps: %.4f\n' % (sub_iou, obj_iou, rpn_results["gpi_overlaps"] / epoch_iter))
                        f.write('acc: %.4f\n' % (acc))
                        f.write('recall0: %.4f recall1: %.4f recall2: %.4f recall3: %.4f\n' % (recall0, recall1, recall2, recall3))
                        f.write('prec0: %.4f prec1: %.4f prec2: %.4f prec3: %.4f\n' % (prec0, prec1, prec2, prec3))
                        f.write('attention neighbour\n')
                        f.write('-------------------\n')
                        f.write(str(gpi_attention["neighbour"]) + "\n")
                        f.write('attention nodes\n')
                        f.write('-------------------\n')
                        f.write(str(gpi_attention["node"])+ "\n")
                        img_node_atten = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                        border = np.argsort(gpi_attention["node"])[::-1]
                        for bb_i in range(min(7, pred_boxes.shape[0])):
                            bb = border[bb_i]
                            cv2.rectangle(img_node_atten, (int(pred_boxes[bb, 0]), int(pred_boxes[bb, 1])), (int(pred_boxes[bb, 2]), int(pred_boxes[bb, 3])), color[bb % len(color) ], 5)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img_node_atten,str(bb_i),(int(pred_boxes[bb, 0]+5), int(pred_boxes[bb, 1]+25)), font, 1,(255,255,255),2,cv2.LINE_AA)
                        cv2.imwrite(os.path.join(path, "img_node_atten.jpg"), img_node_atten)

                        # scene-graph
                        f.write('objects\n')
                        f.write('-------\n')
                        ent_score = predictions['ent_cls_score']
                        ent_probe = softmax(ent_score)
                        ent_probe2 = np.max(ent_probe, axis=1) 
                        f.write(str(ent_probe2)+ "\n")
                        ent_pred = np.argmax(ent_probe, axis=1)
                        ent_pred_w = [objs_cat[ent_pred[i]] for i in range(ent_pred.shape[0])]
                        for i in range(ent_pred.shape[0]):
                            if ent_probe2[i] < 0.5 or predictions["cls_pred_gpi"][0][i] == 0:
                                ent_pred_w[i] = "none"
                        f.write(str(ent_pred_w) + "\n")
                        
                        img_pred = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                        for bb in range(pred_boxes.shape[0]):
                            if ent_pred_w[bb] != "none":
                                cv2.rectangle(img_pred, (int(pred_boxes[bb, 0]), int(pred_boxes[bb, 1])), (int(pred_boxes[bb, 2]), int(pred_boxes[bb, 3])), color[bb % len(color)], 5)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(img_pred, str(bb) + ") " + ent_pred_w[bb],(int(pred_boxes[bb, 0]+5), int(pred_boxes[bb, 1]+25)), font, 1,(255,255,255),2,cv2.LINE_AA)
                        cv2.imwrite(os.path.join(path, "img_ent_preds.jpg"), img_pred)
                        

                        f.write('relations\n')
                        f.write('-------\n')
                        rel_score = predictions['rel_cls_score']
                        rel_probe = softmax(rel_score)
                        rel_probe2 = np.max(rel_probe, axis=2) 
                        f.write(str(rel_probe2)+ "\n")
                        rel_pred = np.argmax(rel_probe, axis=2)
                        #rel_pred_w = [rels_cat[rel_pred[i]] for i in range(ent_pred.shape[0])]
                        rel_pred_w = [[None] * ent_pred.shape[0]] * ent_pred.shape[0]
                        for i in range(ent_pred.shape[0]):
                            for j in range(ent_pred.shape[0]):
                                if ent_pred_w[i] != "none" and ent_pred_w[j] != "none" and rel_probe2[i,j] > 0.7:
                                #if np.sum(proposal_targets['partial_relation_class'][i,j]) != 0 and proposal_targets['labels_mask'][i] == 1.0 and proposal_targets['labels_mask'][j] == 1.0:
                                    rel_pred_w[i][j] = rels_cat[rel_pred[i,j]]
                                    f.write("(" + str(i) + ", " + str(j) + ") -- > " + "(" + ent_pred_w[i] + ", " + rel_pred_w[i][j] + "," + ent_pred_w[j] + ") - " + str(rel_probe2[i,j]) + "\n")
                        
                        
                        #img_pred = blobs["data"][0].copy() + cfg.PIXEL_MEANS
                        #for bb in range(pred_boxes.shape[0]):
                        #    if rel_pred_w[bb] != "none":
                        #        cv2.rectangle(img_pred, (int(pred_boxes[bb, 0]), int(pred_boxes[bb, 1])), (int(pred_boxes[bb, 2]), int(pred_boxes[bb, 3])), color[bb % len(color)], 5)
                        #        font = cv2.FONT_HERSHEY_SIMPLEX
                        #        cv2.putText(img_pred, ent_pred_w[bb],(int(pred_boxes[bb, 0]+5), int(pred_boxes[bb, 1]+25)), font, 1,(255,255,255),2,cv2.LINE_AA)
                        #cv2.imwrite(os.path.join(path, "img_ent_preds.jpg"), img_pred)
                        

                        print("saved")
                

                epoch_iter += 1.0



            except Exception as e:
                print(e)
                print("error iter %d" % (int(epoch_iter)))
                continue

            timer.toc()

            # Display training information
            if (epoch == 1 and int(epoch_iter) % (cfg.TRAIN.DISPLAY) == 0) or (int(epoch_iter) % (10000) == 0):
                print_stat(name, epoch, epoch_iter, lr, accum_results, accum_losses)
          



def print_stat(name, epoch, epoch_iter, lr, accum_results, losses):
    sub_iou = float(accum_results['sub_iou']) / accum_results['total']
    obj_iou = float(accum_results['obj_iou']) / accum_results['total']
    sub_kl = float(accum_results['sub_kl']) / accum_results['total']
    obj_kl = float(accum_results['obj_kl']) / accum_results['total']

    acc = float(accum_results['acc']) / accum_results['total']
    prec0 = float(accum_results['prec0']) / (accum_results['prec0_total'] + 1.0)
    prec1 = float(accum_results['prec1']) / (accum_results['prec1_total'] + 1.0)
    prec2 = float(accum_results['prec2']) / (accum_results['prec2_total'] + 1.0)
    prec3 = float(accum_results['prec3']) / (accum_results['prec3_total'] + 1.0)
    recall0 = float(accum_results['recall0']) / (accum_results['recall0_total'] + 1.0)
    recall1 = float(accum_results['recall1']) / (accum_results['recall1_total'] + 1.0)
    recall2 = float(accum_results['recall2']) / (accum_results['recall2_total'] + 1.0)
    recall3 = float(accum_results['recall3']) / (accum_results['recall3_total'] + 1.0)
    try:   
        f10 = 2 * recall0 * prec0 / (recall0 + prec0)
        f11 = 2 * recall1 * prec1 / (recall1 + prec1)
        f12 = 2 * recall2 * prec2 / (recall2 + prec2)
        f13 = 2 * recall3 * prec3 / (recall3 + prec3)
    except:
        f10 = 0
        f11 = 0
        f12 = 0
        f13 = 0

    sub_iou_bl = float(accum_results['sub_iou_bl']) / accum_results['total_bl']
    obj_iou_bl = float(accum_results['obj_iou_bl']) / accum_results['total_bl']
    sub_kl_bl = float(accum_results['sub_kl_bl']) / accum_results['total_bl']
    obj_kl_bl = float(accum_results['obj_kl_bl']) / accum_results['total_bl']

    acc_bl = float(accum_results['acc_bl']) / accum_results['total_bl']
    prec0_bl = float(accum_results['prec0_bl']) / (accum_results['prec0_total_bl'] + 1.0)
    prec1_bl = float(accum_results['prec1_bl']) / (accum_results['prec1_total_bl'] + 1.0)
    prec2_bl = float(accum_results['prec2_bl']) / (accum_results['prec2_total_bl'] + 1.0)
    prec3_bl = float(accum_results['prec3_bl']) / (accum_results['prec3_total_bl'] + 1.0)
    recall0_bl = float(accum_results['recall0_bl']) / (accum_results['recall0_total_bl'] + 1.0)
    recall1_bl = float(accum_results['recall1_bl']) / (accum_results['recall1_total_bl'] + 1.0)
    recall2_bl = float(accum_results['recall2_bl']) / (accum_results['recall2_total_bl'] + 1.0)
    recall3_bl = float(accum_results['recall3_bl']) / (accum_results['recall3_total_bl'] + 1.0)
    try:
        f10_bl = 2 * recall0_bl * prec0_bl / (recall0_bl + prec0_bl)
        f11_bl = 2 * recall1_bl * prec1_bl / (recall1_bl + prec1_bl)
        f12_bl = 2 * recall2_bl * prec2_bl / (recall2_bl + prec2_bl)
        f13_bl = 2 * recall3_bl * prec3_bl / (recall3_bl + prec3_bl)
    except:
        f10_bl = 0
        f11_bl = 0
        f12_bl = 0
        f13_bl = 0
    print('\n###### %s (%s): epoch %d iter: %d, total loss: %.4f lr: %f' % \
          (name, cfg.TRAIN.SNAPSHOT_PREFIX, epoch, int(epoch_iter), losses["total_loss"] / epoch_iter, lr.eval()))
    print('###scene-graph')
    print(">>> loss_entity_gt: %.6f loss_relation_gt: %.6f acc_entity_gt: %.4f acc_relation_gt %.4f" % (losses["ent_cross_entropy0"] / epoch_iter, losses["rel_cross_entropy0"] / epoch_iter, accum_results["gt_sg_entity_acc"] / epoch_iter, accum_results["gt_sg_relation_acc"] / epoch_iter))
    print(">>> loss_entity: %.6f loss_relation: %.6f acc_entity: %.4f acc_relation %.4f" % (losses["ent_cross_entropy"] / epoch_iter, losses["rel_cross_entropy"] / epoch_iter, accum_results["sg_entity_acc"] / epoch_iter, accum_results["sg_relation_acc"] / epoch_iter))
    print('###rpn')
    print('>>> rpn_loss_cls: %.6f rpn_loss_box: %.6f loss_box: %.6f' % (losses["rpn_cross_entropy"] / epoch_iter, losses["rpn_loss_box"] / epoch_iter, losses["loss_box"] / epoch_iter))
    print('###gpi loss: %.6f' % (losses["cross_entropy_gpi"] / epoch_iter))
    print('sub_iou: %.4f obj_iou: %.4f sub_kl: %.4f obj_kl: %.4f rpn_overlaps: %.4f' % (sub_iou, obj_iou, sub_kl, obj_kl, accum_results["gpi_overlaps"] / epoch_iter))
    print('acc: %.4f ' % (acc))
    print('recall0: %.4f recall1: %.4f recall2: %.4f recall3: %.4f' % (recall0, recall1, recall2, recall3))
    print('prec0: %.4f prec1: %.4f prec2: %.4f prec3: %.4f' % (prec0, prec1, prec2, prec3))
    print('f10: %.4f f11: %.4f f12: %.4f f13: %.4f' % (f10, f11, f12, f13))

    print('###baseline loss: %.6f' % (losses["cross_entropy_baseline"] / epoch_iter))
    print('sub_iou_bl: %.4f obj_iou_bl: %.4f sub_kl_bl: %.4f obj_kl_bl: %.4f rpn_overlaps_bl: %.4f' % (sub_iou_bl, obj_iou_bl, sub_kl_bl, obj_kl_bl, accum_results["baseline_overlaps"] / epoch_iter))
    print('acc_bl: %.4f' % (acc_bl))
    print('recall0_bl: %.4f recall1_bl: %.4f recall2_bl: %.4f recall3_bl: %.4f' % (recall0_bl, recall1_bl, recall2_bl, recall3_bl))
    print('prec0_bl: %.4f prec1_bl: %.4f prec2_bl: %.4f prec3_bl: %.4f' % (prec0_bl, prec1_bl, prec2_bl, prec3_bl))
    print('f10_bl: %.4f f11_bl: %.4f f12_bl: %.4f f13_bl: %.4f' % (f10_bl, f11_bl, f12_bl, f13_bl))

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing data...')
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
    print('Filtered {} roidb entries: {} -> {}' % (num - num_after,
                                                       num, num_after))
    return roidb


def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=100, just_test=False):
    """Train a Faster R-CNN network."""
    #pretrained_model="output/res101/VisualGenome/default/gpir_imagenet_train_val_vgrr_flipped_nsgf_cont_iter_0.ckpt"
    #just_test=True

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir, tb_dir,
                           pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(sess, max_iters,just_test)
        print('done solving')


MASK_WIDTH = 14
MASK_HEIGHT = 14
def softmax(x):
    xexp = np.exp(x)
    return xexp / np.sum(xexp, axis=-1, keepdims=1)
    

def rpn_test(gt_bbox, pred_boxes0, pred_boxes, gt_ent, gt_rel, predictions):
    overlaps0 = bbox_overlaps(np.ascontiguousarray(gt_bbox, dtype=np.float), np.ascontiguousarray(pred_boxes0, dtype=np.float))
    overlaps0_assign = np.argmax(overlaps0, axis=1)
    max_overlaps0 = overlaps0.max(axis=1)

    overlaps = bbox_overlaps(np.ascontiguousarray(gt_bbox, dtype=np.float), np.ascontiguousarray(pred_boxes, dtype=np.float))
    overlaps_assign = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps.max(axis=1)

    pred_ent = predictions['ent_cls_score']
    pred_ent = softmax(pred_ent[overlaps_assign])
    ent_accuracy = np.sum(np.multiply(pred_ent, gt_ent)) / np.sum(gt_ent)
    
    pred_rel = predictions['rel_cls_score']
    pred_rel = softmax(pred_rel[overlaps_assign,:][:,overlaps_assign]) 
    rel_accuracy = np.sum(np.multiply(pred_rel, gt_rel)) / np.sum(gt_rel)
    
    pred_ent0 = predictions['ent_cls_score0']
    pred_ent0 = softmax(pred_ent0)
    ent0_accuracy = np.sum(np.multiply(pred_ent0, gt_ent)) / np.sum(gt_ent)
    
    pred_rel0 = predictions['rel_cls_score0']
    pred_rel0 = softmax(pred_rel0)
    rel0_accuracy = np.sum(np.multiply(pred_rel0, gt_rel)) / np.sum(gt_rel)

    results = {}
    results["baseline_overlaps"] = np.mean(max_overlaps0)
    results["gpi_overlaps"] = np.mean(max_overlaps)
    results["sg_entity_acc"] = ent_accuracy
    results["gt_sg_entity_acc"] = ent0_accuracy    
    results["sg_relation_acc"] = rel_accuracy
    results["gt_sg_relation_acc"] = rel0_accuracy
    
    return results

def iou_test(gt, gt_bbox, pred_mask, pred_label, pred, pred_prob, pred_bbox, im_info):
    results = {}
    # number of objects
    results["total"] = 1

    results["sub_iou"] = 0.0
    results["obj_iou"] = 0.0
    results["sub_kl"] = 0.0
    results["obj_kl"] = 0.0
    
    # accuracy
    results["acc"] = np.sum(pred == pred_label).astype(float) / pred_label.shape[0] 
   
    # precision
    for i in range(4):
        total = np.sum(np.logical_and(pred == i, pred_mask != 0)).astype(float)
        if total != 0:
            results["prec" + str(i)] = np.sum(np.logical_and(np.logical_and(pred == pred_label, pred_label == i), pred_mask != 0)).astype(float) / total
            results["prec" + str(i) + "_total"] = 1.0
        else:
            results["prec" + str(i)] = 0.0
            results["prec" + str(i) + "_total"] = 0.0

    # recall
    for i in range(4):
        total = np.sum(np.logical_and(pred_label == i, pred_mask != 0)).astype(float)
        if total != 0:
            results["recall" + str(i)] = np.sum(np.logical_and(np.logical_and(pred == pred_label, pred_label == i), pred_mask != 0)).astype(float) / total
            results["recall" + str(i) + "_total"] = 1.0
        else:
            results["recall" + str(i)] = 0.0
            results["recall" + str(i) + "_total"] = 0.0

    
    width = MASK_WIDTH
    height = MASK_HEIGHT
    MASK_SHAPE = (width, height) 
    mask_sub_gt = np.zeros(MASK_SHAPE, dtype=float)
    mask_obj_gt = np.zeros(MASK_SHAPE, dtype=float)
    mask_sub_pred = np.zeros(MASK_SHAPE, dtype=float)
    mask_obj_pred = np.zeros(MASK_SHAPE, dtype=float)
    mask_sub_pred_bool = np.zeros(MASK_SHAPE, dtype=bool)
    mask_obj_pred_bool = np.zeros(MASK_SHAPE, dtype=bool)

    # sub and obj bool mask
    i = np.argmax(pred_prob[:, 1])
    mask_sub_pred_bool[int(math.floor(pred_bbox[i][0] * MASK_SHAPE[0])):int(math.ceil(pred_bbox[i][2] * MASK_SHAPE[0])),
    int(math.floor(pred_bbox[i][1] * MASK_SHAPE[1])):int(math.ceil(pred_bbox[i][3] * MASK_SHAPE[1]))] = True
    i = np.argmax(pred_prob[:, 2])
    mask_obj_pred_bool[int(math.floor(pred_bbox[i][0] * MASK_SHAPE[0])):int(math.ceil(pred_bbox[i][2] * MASK_SHAPE[0])),
    int(math.floor(pred_bbox[i][1] * MASK_SHAPE[1])):int(math.ceil(pred_bbox[i][3] * MASK_SHAPE[1]))] = True
    for i in range(pred.shape[0]):
        if pred[i] == 1:
            x1 = int(math.floor(pred_bbox[i][0] * MASK_SHAPE[0]))
            x2 = int(math.ceil(pred_bbox[i][2] * MASK_SHAPE[0]))
            y1 = int(math.floor(pred_bbox[i][1] * MASK_SHAPE[1]))
            y2 = int(math.ceil(pred_bbox[i][3] * MASK_SHAPE[1]))
            mask_sub_pred_bool[x1:x2, y1:y2] = True
        if pred[i] == 2:
            x1 = int(math.floor(pred_bbox[i][0] * MASK_SHAPE[0]))
            x2 = int(math.ceil(pred_bbox[i][2] * MASK_SHAPE[0]))
            y1 = int(math.floor(pred_bbox[i][1] * MASK_SHAPE[1]))
            y2 = int(math.ceil(pred_bbox[i][3] * MASK_SHAPE[1]))
            mask_obj_pred_bool[x1:x2, y1:y2] = True

    # GT mask
    for i in range(gt.shape[0]):
        if gt[i] == 1:
            x1 = int(math.floor(gt_bbox[i][0] * MASK_SHAPE[0]))
            x2 = int(math.ceil(gt_bbox[i][2] * MASK_SHAPE[0]))
            y1 = int(math.floor(gt_bbox[i][1] * MASK_SHAPE[1]))
            y2 = int(math.ceil(gt_bbox[i][3] * MASK_SHAPE[1]))
            mask_sub_gt[x1:x2, y1:y2] = 1.0
        if gt[i] == 2:
            x1 = int(math.floor(gt_bbox[i][0] * MASK_SHAPE[0]))
            x2 = int(math.ceil(gt_bbox[i][2] * MASK_SHAPE[0]))
            y1 = int(math.floor(gt_bbox[i][1] * MASK_SHAPE[1]))
            y2 = int(math.ceil(gt_bbox[i][3] * MASK_SHAPE[1]))
            mask_obj_gt[x1:x2, y1:y2] = 1.0

    # predicted mask
    for i in range(pred.shape[0]):
        x1 = int(math.floor(pred_bbox[i][0] * MASK_SHAPE[0]))
        x2 = int(math.ceil(pred_bbox[i][2] * MASK_SHAPE[0]))
        y1 = int(math.floor(pred_bbox[i][1] * MASK_SHAPE[1]))
        y2 = int(math.ceil(pred_bbox[i][3] * MASK_SHAPE[1]))
        mask = np.zeros(MASK_SHAPE, dtype=float)
        mask[x1:x2, y1:y2] = 1.0
        mask_sub_pred = np.maximum(mask_sub_pred, mask * pred_prob[i, 1])
        mask_obj_pred = np.maximum(mask_obj_pred, mask * pred_prob[i, 2])

    sub_iou = iou(mask_sub_gt.astype(bool), mask_sub_pred_bool)
    obj_iou = iou(mask_obj_gt.astype(bool), mask_obj_pred_bool)
    sub_kl = kl(mask_sub_gt, mask_sub_pred)
    obj_kl = kl(mask_obj_gt, mask_obj_pred)


    results["sub_iou"] += sub_iou
    results["obj_iou"] += obj_iou
    results["sub_kl"] += sub_kl
    results["obj_kl"] += obj_kl

    return results


def iou(mask_a, mask_b):
    union = np.sum(np.logical_or(mask_a, mask_b))
    if union == 0:
        return 0.0
    intersection = np.sum(np.logical_and(mask_a, mask_b))
    return float(intersection) / float(union)

def kl(mask_gt, mask_pred):
    gt = mask_gt.astype(float) / (np.sum(mask_gt) + k.backend.epsilon())
    pred = mask_pred.astype(float) / (np.sum(mask_pred) + k.backend.epsilon())
    x = np.log(k.backend.epsilon() + gt/(pred + k.backend.epsilon()))
    return np.sum(x * gt)


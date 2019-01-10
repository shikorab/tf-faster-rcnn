import tensorflow as tf

class Gpi(object):
    def __init__(self, nof_ent_classes, nof_rel_classes, nof_node_features=1024, nof_relation_features=1024, rnn_steps=1, layers = [516, 516], gpi_type="FeatureAttention", sg_as_features=False, phase_ph=None):
        self.nof_node_features = nof_node_features
        self.nof_relation_features = nof_relation_features
        self.nof_ent_classes = nof_ent_classes
        self.nof_rel_classes = nof_rel_classes
        self.nof_rnn_steps = rnn_steps
        self.gpi_type = gpi_type
        self.layers = layers
        self.activation_fn = tf.nn.relu
        self.reuse = None
        self.sg_as_features = sg_as_features
        self.phase_ph = phase_ph
        self.attention = {}

    def predict(self, node_features, relation_features, gt_node_features, gt_relation_features):
        self.reuse = False
        pred_node_features = node_features
        for step in range(self.nof_rnn_steps):
            pred_node_features, graph_features, ent_score, rel_score, ent_score0, rel_score0 = \
                self.gpi_step(relation_features=relation_features,
                         node_features=pred_node_features,
                         gt_node_features=gt_node_features, gt_relation_features=gt_relation_features,
                         scope="gpi")
            self.reuse = True

        return pred_node_features, ent_score, rel_score, ent_score0, rel_score0 #;tf.concat((node_features, pred_node_features), axis=1)

    def nn(self, features, layers, out, scope_name, last_activation=None):
        with tf.variable_scope(scope_name):
            index = 0
            h = tf.concat(features, axis=-1)
            #h = tf.contrib.layers.dropout(h, keep_prob=0.9, is_training=self.phase_ph)
            for layer in layers:
                scope = str(index)
                h = tf.contrib.layers.fully_connected(h, layer, reuse=self.reuse, scope=scope,
                                                      activation_fn=self.activation_fn)
                #h = tf.contrib.layers.dropout(h, keep_prob=0.9, is_training=self.phase_ph)
                index += 1

            scope = str(index)
            y = tf.contrib.layers.fully_connected(h, out, reuse=self.reuse, scope=scope, activation_fn=last_activation)
        return y

    def gpi_step(self, node_features, relation_features, gt_node_features, gt_relation_features, scope):
        with tf.variable_scope(scope):
            
            if self.sg_as_features:
                ent_score = node_features
                ent_score0 = gt_node_features
                rel_score = relation_features
                rel_score0 = gt_relation_features
                node_features = tf.stop_gradient(node_features)
                relation_features = tf.stop_gradient(relation_features)

            N = tf.slice(tf.shape(relation_features), [0], [1], name="N")

            # expand object confidence
            self.expand_node_shape = tf.concat((N, tf.shape(node_features)), 0)
            self.expand_object_features = tf.add(tf.zeros(self.expand_node_shape), node_features)
            # expand subject confidence
            self.expand_subject_features = tf.transpose(self.expand_object_features, perm=[1, 0, 2])

            ##
            # Node Neighbours
            self.object_ngbrs = [self.expand_object_features, self.expand_subject_features, relation_features]
            # apply phi
            self.object_ngbrs_phi = self.nn(features=self.object_ngbrs, layers=self.layers,
                                            last_activation=self.activation_fn, out=self.nof_node_features, scope_name="nn_phi")
            # Attention mechanism
            if self.gpi_type == "FeatureAttention":
                self.object_ngbrs_scores = self.nn(features=self.object_ngbrs, layers=self.layers, out=self.nof_node_features,
                                                   scope_name="nn_phi_atten")
                self.object_ngbrs_weights = tf.nn.softmax(self.object_ngbrs_scores, dim=1)
                self.attention["neighbour"] = tf.reduce_mean(self.object_ngbrs_weights, axis=2) 
                self.object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(self.object_ngbrs_phi, self.object_ngbrs_weights),
                                                          axis=1)

            elif self.gpi_type == "NeighbourAttention":
                self.object_ngbrs_scores = self.nn(features=self.object_ngbrs, layers=[self.nof_node_features], out=1,
                                                   scope_name="nn_phi_atten")
                self.object_ngbrs_weights = tf.nn.softmax(self.object_ngbrs_scores, dim=1)
                self.attention["neighbour"] = tf.squeeze(self.object_ngbrs_weights) 
                
                self.object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(self.object_ngbrs_phi, self.object_ngbrs_weights),
                                                          axis=1)
            else:
                self.object_ngbrs_phi_all = tf.reduce_sum(self.object_ngbrs_phi, axis=1) / tf.constant(64.0)

            ##
            # Nodes
            self.object_ngbrs2 = [node_features, self.object_ngbrs_phi_all]
            # apply alpha
            self.object_ngbrs2_alpha = self.nn(features=self.object_ngbrs2, layers=self.layers, out=self.nof_node_features,
                                               last_activation=self.activation_fn, scope_name="nn_phi2")
            # Attention mechanism
            if self.gpi_type == "FeatureAttention" or self.gpi_type == "Linguistic":
                self.object_ngbrs2_scores = self.nn(features=self.object_ngbrs2, layers=self.layers, out=self.nof_node_features,
                                                    scope_name="nn_phi2_atten")
                self.object_ngbrs2_weights = tf.nn.softmax(self.object_ngbrs2_scores, dim=0)
                self.object_ngbrs2_alpha_all = tf.reduce_sum(
                    tf.multiply(self.object_ngbrs2_alpha, self.object_ngbrs2_weights), axis=0)
                self.attention["node"] = tf.reduce_mean(self.object_ngbrs2_weights, axis=1)
            elif self.gpi_type == "NeighbourAttention":
                self.object_ngbrs2_scores = self.nn(features=self.object_ngbrs2, layers=[self.nof_node_features], out=1,
                                                    scope_name="nn_phi2_atten")
                self.object_ngbrs2_weights = tf.nn.softmax(self.object_ngbrs2_scores, dim=0)
                self.object_ngbrs2_alpha_all = tf.reduce_sum(
                    tf.multiply(self.object_ngbrs2_alpha, self.object_ngbrs2_weights), axis=0)
                self.attention["node"] = tf.squeeze(self.object_ngbrs2_weights)
            else:
                self.object_ngbrs2_alpha_all = tf.reduce_sum(self.object_ngbrs2_alpha, axis=0) / tf.constant(64.0)

            expand_graph_shape = tf.concat((N, N, tf.shape(self.object_ngbrs2_alpha_all)), 0)
            expand_graph = tf.add(tf.zeros(expand_graph_shape), self.object_ngbrs2_alpha_all)
            ##
            # rho entity (entity prediction)
            # The input is entity features, entity neighbour features and the representation of the graph
            self.object_all_features = [node_features, expand_graph[0], self.object_ngbrs_phi_all]
            obj_delta = self.nn(features=self.object_all_features, layers=self.layers, out=self.nof_node_features, scope_name="nn_obj")
            obj_forget_gate = self.nn(features=self.object_all_features, layers=self.layers, out=self.nof_node_features, scope_name="nn_obj_forgate", last_activation=tf.nn.sigmoid)
            pred_node_features = obj_delta# + obj_forget_gate * node_features
            
            if not self.sg_as_features:
                ##
                # relation score
                self.relation_all_features = [self.expand_object_features, self.expand_subject_features, relation_features, expand_graph]
                rel_score = self.nn(features=self.relation_all_features, layers=[], out=self.nof_rel_classes, scope_name="ent_score")
                ##
                # entity score
                ent_score = self.nn(features=[pred_node_features], layers=[], out=self.nof_ent_classes, scope_name="rel_score")
            
                ##
                # relation score
                N = tf.slice(tf.shape(gt_relation_features), [0], [1], name="N")

                # expand object confidence
                self.gt_expand_node_shape = tf.concat((N, tf.shape(gt_node_features)), 0)
                self.gt_expand_object_features = tf.add(tf.zeros(self.gt_expand_node_shape), gt_node_features)
                # expand subject confidence
                self.gt_expand_subject_features = tf.transpose(self.gt_expand_object_features, perm=[1, 0, 2])

                self.relation_all_features = [gt_relation_features, self.gt_expand_object_features, self.gt_expand_subject_features]
                rel_score0 = self.nn(features=self.relation_all_features, layers=[], out=self.nof_rel_classes, scope_name="rel_score0")
                ##
                # entity score
                ent_score0 = self.nn(features=[gt_node_features], layers=[], out=self.nof_ent_classes, scope_name="ent_score0")
            
            return pred_node_features, expand_graph, ent_score, rel_score, ent_score0, rel_score0

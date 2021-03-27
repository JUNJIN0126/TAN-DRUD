import tensorflow as tf
import numpy as np
from rank_metrics import rank_eval
import argparse
from tensorflow.contrib import rnn

def ill_cal(pred, sl):
    nll = 0
    cur_pos = 0
    for i in range(len(sl)):
        length = sl[i]
        cas_nll = pred[cur_pos : cur_pos+length]
        cur_pos += length
        nll += (np.sum(cas_nll)/float(length))
    return nll

# cas_emb:[b,n,d]  cas_mask:[b,n,1]

def tan(cas_emb, cas_emb1, graph_emb, cas_mask, time_weight, hidden_size, keep_prob):
    cas_encoding = user2user(cas_emb, cas_emb1,graph_emb,cas_mask, hidden_size, keep_prob)     # [b,n,d]
    gra_encoding = gra2user(tf.cast(graph_emb,tf.float32), cas_mask,hidden_size, keep_prob)
    cas_encoding = gra_encoding + cas_encoding

    return user2cas(cas_encoding, cas_mask, time_weight, hidden_size, keep_prob)

def gra2user(graph_emb, cas_mask,hidden_size, keep_prob):
    with tf.variable_scope('gra2user'):
        gra_hidden = dense(graph_emb,hidden_size,tf.nn.elu,keep_prob,'graph_trans') * cas_mask
        return gra_hidden


def user2user(cas_emb,cas_emb1, gra_emb,cas_mask, hidden_size, keep_prob):
    with tf.variable_scope('user2user'):
        
        g_norm = tf.sqrt(tf.reduce_mean(tf.square(gra_emb),axis=2,keep_dims=True))
        g_sim = tf.matmul(gra_emb,tf.transpose(gra_emb,[0,2,1]))
        g_dis_score = g_sim / (tf.matmul(g_norm,tf.transpose(g_norm,[0,2,1])) + 1e-8)


        bs, sl = tf.shape(cas_emb)[0], tf.shape(cas_emb)[1] 
        col, row = tf.meshgrid(tf.range(sl), tf.range(sl))            # [n,n]
        direction_mask1 = tf.greater(row, col)
        direction_mask2 = tf.greater(col, row) # [n,n]
        direction_mask_tile1 = tf.tile(tf.expand_dims(direction_mask1, 0), [bs, 1, 1])     # [b,n,n]
        direction_mask_tile2 = tf.tile(tf.expand_dims(direction_mask2, 0), [bs, 1, 1])
        length_mask_tile = tf.tile(tf.expand_dims(tf.squeeze(tf.cast(cas_mask,tf.bool),-1), 1), [1, sl, 1])    #
        attention_mask1 = tf.cast(tf.logical_and(direction_mask_tile1, length_mask_tile), tf.float32)         # [b,n,n]
        attention_mask2 = tf.cast(tf.logical_and(direction_mask_tile2, length_mask_tile), tf.float32)         # [b,n,n]
        cas_hidden = dense(cas_emb, hidden_size, tf.nn.elu, keep_prob, 'hidden')* cas_mask# [b,n,d] *cas_mask
  
        cas_hidden1 = dense(cas_emb1, hidden_size, tf.nn.elu, keep_prob, 'hidden1') * cas_mask   # [b,n,d]

        head = dense(cas_hidden, hidden_size, tf.identity, keep_prob, 'head', False) # [b,n,d]
        tail = dense(cas_hidden1, hidden_size, tf.identity, keep_prob, 'tail', False) # [b,n,d]
        head1 = dense(cas_hidden1, hidden_size, tf.identity, keep_prob, 'head1', False)  # [b,n,d]
        tail1= dense(cas_hidden, hidden_size, tf.identity, keep_prob, 'tail1', False)  # [b,n,d]

        matching_logit1 = tf.matmul(head, tf.transpose(tail,perm=[0,2,1])) + (1-attention_mask1) * (-1e30) #
        attention_score1 = tf.nn.softmax(matching_logit1, -1) * attention_mask1 #

        g_dis_score1 = tf.cast(g_dis_score,tf.float32) * attention_mask1
        attention_score1 = tf.nn.softmax(attention_score1 * g_dis_score1, -1)



        depend_emb1 = tf.matmul(attention_score1, cas_hidden)         # [b,n,d]
        depend_emb1 = depend_emb1 + cas_hidden
        matching_logit2 = tf.matmul(head1, tf.transpose(tail1,perm=[0,2,1])) + (1-attention_mask2) * (-1e30) #
        attention_score2 = tf.nn.softmax(matching_logit2, -1) * attention_mask2 #

        g_dis_score2 = tf.cast(g_dis_score,tf.float32) * attention_mask2
        attention_score2 = tf.nn.softmax(attention_score2 * g_dis_score2, -1)

        depend_emb2 = tf.matmul(attention_score2, cas_hidden1)         # [b,n,d]
        depend_emb2 = depend_emb2 + cas_hidden1
        fusion_gate1 = dense(tf.concat([depend_emb1, depend_emb2], 2), hidden_size, tf.sigmoid, keep_prob, 'fusion_gate1')  # [b,n,d]
        #depend_emb =  (fusion_gate1*depend_emb1 + (1-fusion_gate1)*depend_emb2)
        #fusion_gate = dense(tf.concat([cas_hidden, depend_emb], 2), hidden_size, tf.sigmoid, keep_prob, 'fusion_gate')
        fusion_gate2 = dense(tf.concat([depend_emb1, depend_emb2], 2), hidden_size, tf.sigmoid, keep_prob,'fusion_gate2')
        return ((1-fusion_gate1)*depend_emb1+(1-fusion_gate2)*depend_emb2)*cas_mask

        #eturn (fusion_gate*cas_hidden + (1-fusion_gate)*depend_emb) * cas_mask # # [b,n,d]

def user2cas(cas_encoding, cas_mask, time_weight, hidden_size, keep_prob):
    with tf.variable_scope('user2cas'):
        map1 = dense(cas_encoding, hidden_size, tf.nn.elu, keep_prob, 'map1')   # [b,n,d]
        time_influence = dense(time_weight, hidden_size, tf.nn.elu, keep_prob, 'time_influence')
        map2 = dense(map1 * time_influence, 1, tf.identity, keep_prob, 'map2')
        attention_score =  tf.nn.softmax(map2 + (-1e30) * (1 - cas_mask) , 1) * cas_mask
        return tf.reduce_sum(attention_score * cas_encoding, 1), attention_score



def dense(input, out_size, activation, keep_prob, scope, need_bias=True):
    with tf.variable_scope(scope):
        W = tf.get_variable('W', [input.get_shape()[-1], out_size], dtype=tf.float32)
        b = tf.get_variable('b', [out_size], tf.float32, tf.zeros_initializer(), trainable=need_bias)
        flatten = tf.matmul(tf.reshape(input, [-1, tf.shape(input)[-1]]), W) + b
        out_shape = [tf.shape(input)[i] for i in range(len(input.get_shape())-1)] + [out_size]
        return tf.nn.dropout(activation(tf.reshape(flatten, out_shape)), keep_prob)

class Model(object):
    def __init__(self, config):
        self.num_nodes = config.num_nodes
        print(self.num_nodes)
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.learning_rate = config.learning_rate
        self.l2_weight = config.l2_weight
        self.train_dropout = config.dropout
        self.n_time_interval = config.n_time_interval
        self.optimizer = config.optimizer
        self.g = np.load(config.data_name + "_embedding.npy")

    def build_model(self):
        with tf.variable_scope("model",initializer=tf.contrib.layers.xavier_initializer()) as scope:
            self.cas = tf.placeholder(tf.int32, [None, None])                    # (b,n)

            self.cas_length= tf.reduce_sum(tf.sign(self.cas),1)
            self.cas_mask = tf.expand_dims(tf.sequence_mask(self.cas_length, tf.shape(self.cas)[1], tf.float32), -1)
            self.dropout = tf.placeholder(tf.float32)
            self.labels = tf.placeholder(tf.int32, [None])                          # (b,)

            self.time_interval_index = tf.placeholder(tf.int32, [None, None])       # (b,n)

            self.num_cas = tf.placeholder(tf.float32)

            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable(
                    "embedding",[self.num_nodes,self.embedding_size],  dtype=tf.float32)
                self.embedding1 = tf.get_variable(
                    "embedding1", [self.num_nodes,self.embedding_size], dtype=tf.float32)
                self.graph_embedding = tf.constant(self.g)
                self.cas_emb = tf.nn.embedding_lookup(self.embedding, self.cas)  # (b,n,l)
                self.cas_emb1 = tf.nn.embedding_lookup(self.embedding1, self.cas)
                self.graph_emb = tf.nn.embedding_lookup(self.graph_embedding, self.cas)
                self.time_lambda = tf.get_variable('time_lambda', [self.n_time_interval+1, self.hidden_size], dtype=tf.float32) #,
                self.time_weight = tf.nn.embedding_lookup(self.time_lambda, self.time_interval_index)

            with tf.variable_scope("tan") as scope:
            	
                (self.tan,self.attention)= tan(self.cas_emb,self.cas_emb1 , self.graph_emb, self.cas_mask, self.time_weight, self.hidden_size, self.dropout)
             	
            with tf.variable_scope("loss"):
                
                l0 = self.tan
                self.logits = dense(l0, self.num_nodes, tf.identity, 1.0, 'logits')
                self.nll = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, self.num_nodes, dtype=tf.float32), logits=self.logits)
                self.loss = tf.reduce_mean(self.nll,-1)
                for v in tf.trainable_variables():
                    self.loss += self.l2_weight * tf.nn.l2_loss(v)
                if self.optimizer == 'adaelta':
                    self.train_op = tf.train.AdadeltaOptimizer(self.learning_rate, rho=0.999).minimize(self.loss)
                else:
                    self.train_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.99).minimize(self.loss)

    def train_batch(self, sess, batch_data):
        cas, next_user, time_interval_index, seq_len = batch_data
        feed = {self.cas: cas,
                self.labels: next_user,
                self.dropout: self.train_dropout,
                self.time_interval_index: time_interval_index,
                self.num_cas: len(seq_len)
               }
        _, _, nll = sess.run([self.train_op, self.loss, self.nll], feed_dict = feed)
        batch_nll = np.sum(nll)
        return batch_nll

    def test_batch(self, sess, batch_test):
        cas, next_user, time_interval_index, seq_len = batch_test
        feed = {self.cas: cas,
                self.labels: next_user,
                self.time_interval_index: time_interval_index,
                self.dropout: 1.0
               }
        logits, nll = sess.run([self.logits, self.nll], feed_dict = feed)
        mrr, macc1, macc5, macc10, macc50, macc100 = rank_eval(logits, next_user, seq_len)
        batch_cll = np.sum(nll)
        batch_ill = ill_cal(nll, seq_len)
        return batch_cll, batch_ill, mrr, macc1, macc5, macc10, macc50, macc100
    
    def test_batch_attention(self, sess, batch_test):
        cas, next_user, time_interval_index, seq_len = batch_test
        feed = {self.cas: cas,
                self.labels: next_user,
                self.time_interval_index: time_interval_index,
                self.dropout: 1.0
               }
        attention = sess.run([self.attention],feed_dict=feed)
        return attention

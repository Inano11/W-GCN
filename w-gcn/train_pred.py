from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

# from utils import *
from utils_modify import *
from models import GCN, MLP
import numpy as np
import sklearn.metrics as sm



# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
# seed
# 7:93.0 9:93.2 10:93.3 12:93.1 13:93.1 16:93.2 21:93.3 25:93.3

epoch_set = 10000

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citese                                            er', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.') # 0.01
flags.DEFINE_integer('epochs', epoch_set, 'Number of epochs to train.') # 400

flags.DEFINE_integer('hidden1', 16, '卷积层第一层的output_dim，第二层的input_dim')
flags.DEFINE_float('dropout', 0.7, 'Dropout rate (1 - keep probability).')

# self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
flags.DEFINE_float('weight_decay', 5e-4, '权重衰减，让权重减少到更小的值') # 5e-4
flags.DEFINE_integer('early_stopping', 10000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)


# 预处理特征矩阵:将特征矩阵归一化处理  (coords, values, shape)
features = preprocess_features(features)

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))


# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # 'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant((features[2]), dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)
    # helper variable for sparse dropout
}


# Create model
# model = model_func(placeholders, input_dim=features[2][1], logging=True)
model = model_func(placeholders, input_dim=300, logging=True)


# Initialize session
sess = tf.Session()


# Define model evaluation function
# def evaluate(features, support, labels, mask, placeholders):
#     t_test = time.time()
#     feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
#     outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
#     return outs_val[0], outs_val[1], (time.time() - t_test)

def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    pred = sess.run([model.predict()], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1],  (time.time() - t_test), pred


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration, pred = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # # add --------------------------------------------
    # # Testing
    # test_cost, test_acc, test_duration, pred = evaluate(features, support, y_test, test_mask, placeholders)
    #
    # print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    #       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    # # add --------------------------------------------

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# # Testing
# test_cost, test_acc, test_duration, pred = evaluate(features, support, y_test, test_mask, placeholders)
#
# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

# diy:
label_pred = []
label_true = []

for index, flag in enumerate(test_mask):
    if flag == True:
        label_t = y_test[index].tolist()[1]
        label_true.append(int(label_t))
        label_p = np.argmax(pred[0].tolist()[index]).tolist()
        label_pred.append(int(label_p))


print('true', label_true)
print('pred', label_pred)


Accuracy = sm.accuracy_score(label_true, label_pred)
print("accuracy", Accuracy)


len_testt = len(label_true)
len_test  = len(label_pred)

if len_test != len_testt:
    print('error!!!!!!!!!')


# R precision
fenzi = 0
fenmu = 0
for index in range(len_test):
    lab_trp = label_true[index]
    lab_prp = label_pred[index]
    if lab_trp == 1:
        if lab_prp == 1:
            fenzi += 1

fenmu = label_pred.count(1)
precisionR = fenzi / fenmu
print('precision_R', precisionR)


# N precision
fenzi = 0
fenmu = 0
for index in range(len_test):
    lab_tnp = label_true[index]
    lab_pnp = label_pred[index]
    if lab_tnp == 0:
        if lab_pnp == 0:
            fenzi += 1

fenmu = label_pred.count(0)
precisionN = fenzi / fenmu
print('precision_N', precisionN)


# R recall
fenzi = 0
fenmu = 0
for index in range(len_test):
    lab_trr = label_true[index]
    lab_prr = label_pred[index]
    if lab_trr == 1:
        if lab_prr == 1:
            fenzi += 1

fenmu = label_true.count(1)
recallR = fenzi / fenmu
print('recall_R', recallR)


# N recall
fenzi = 0
fenmu = 0
for index in range(len_test):
    lab_tnr = label_true[index]
    lab_pnr = label_pred[index]
    if lab_tnr == 0:
        if lab_pnr == 0:
            fenzi += 1

fenmu = label_true.count(0)
recallN = fenzi / fenmu
print('recall_N', recallN)


# R F1
RF1 = 2 * precisionR * recallR / (precisionR + recallR)
print('F1_R', RF1)


# N F1
NF1 = 2 * precisionN * recallN / (precisionN + recallN)
print('F1_N', NF1)

# print('----------copying---------')
# print(Accuracy)
# print(precisionR)
# print(precisionN)
# print(recallR)
# print(recallN)
# print(RF1)
# print(NF1)
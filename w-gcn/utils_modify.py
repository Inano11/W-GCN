import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random

# adj(邻接矩阵)：由于比较稀疏，邻接矩阵格式是LIL的，并且shape为(2708, 2708)
# features（特征矩阵）：每个节点的特征向量也是稀疏的，也用LIL格式存储，features.shape: (2708, 1433)
# labels：ally, ty 数据集叠加构成，labels.shape:(2708, 7)
# train_mask, val_mask, test_mask：shaped都为(2708, )的向量，但是train_mask中的[0,140)范围的是True，其余是False；val_mask中范围为(140, 640]范围为True，其余的是False；test_mask中范围为[1708,2707]范围是True，其余的是False
# y_train, y_val, y_test：shape都是(2708, 7) 。y_train的值为对应与labels中train_mask为True的行，其余全是0；y_val的值为对应与labels中val_mask为True的行，其余全是0；y_test的值为对应与labels中test_mask为True的行，其余全是0
# 特征矩阵进行归一化并返回一个格式为(coords, values, shape)的元组
# 将邻接矩阵加上自环以后，对称归一化，并存储为COO模式，最后返回格式为(coords, values, shape)的元组

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# 数据的读取，这个预处理是把训练集（其中一部分带有标签），测试集，标签的位置，对应的掩码训练标签等返回
# def load_data(dataset_str):
def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).

    以cora为例：
    ind.dataset_str.x => 训练实例的特征向量，是scipy.sparse.csr.csr_matrix类对象，shape:(140, 1433)
    ind.dataset_str.tx => 测试实例的特征向量,shape:(1000, 1433)
    ind.dataset_str.allx => 有标签的+无无标签训练实例的特征向量，是ind.dataset_str.x的超集，shape:(1708, 1433)

    ind.dataset_str.y => 训练实例的标签，独热编码，numpy.ndarray类的实例，是numpy.ndarray对象，shape：(140, 7)
    ind.dataset_str.ty => 测试实例的标签，独热编码，numpy.ndarray类的实例,shape:(1000, 7)
    ind.dataset_str.ally => 对应于ind.dataset_str.allx的标签，独热编码,shape:(1708, 7)

    ind.dataset_str.graph => 图数据，collections.defaultdict类的实例，格式为 {index：[index_of_neighbor_nodes]}
    ind.dataset_str.test.index => 测试实例的id，2157行

    上述文件必须都用python的pickle模块存储
    ————————————————
    版权声明：本文为CSDN博主「yyl424525」的原创文章，遵循 CC 4.0 BY 版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/yyl424525/article/details/100831452
    """
    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))
    #
    # x, y, tx, ty, allx, ally, graph = tuple(objects)
    # test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))

    # test_idx_range = np.sort(test_idx_reorder)
    # with open(r'C:\Users\Inano11\Desktop\GCN_1210\eid_index', 'rb') as f_index:
    #     index_list = pkl.load(f_index)

    # with open(r'C:\Users\Inano11\Desktop\GCN_1210\input_old\feature_matrix_chinese_bert.json', 'rb') as f_fm:
    # with open(r'C:\Users\Inano11\Desktop\GCN_1210\feature_matrix_xlnet.json', 'rb') as f_fm:
    # with open(r'D:\gcn-wxy\gcn_rumor\adj2.json', 'rb') as f_adj:


    hour = 'all'
    seed_num = 2
    seed_train = 3
    weight = 1

    with open(r'E:\GCN_20200211\gcn_pre\GCN_feature\GCN_feature' + str(hour) + str(seed_num) + 'test.json', 'rb') as f_fm:

        features_matrix = pkl.load(f_fm)

    if weight == 1:
        # 有权重的 邻接矩阵
        with open(r'D:\研究生资料2020\GCN_20200211\adj_mt', 'rb') as f_adj:
            adj_matrix = pkl.load(f_adj)

    if weight == 0:
        with open(r'C:\Users\Inano11\Desktop\GCN_1210\input_old\adj_616', 'rb') as f_adj:
            adj_matrix = pkl.load(f_adj)


    # with open(r'E:\GCN_2019\gcn-wxy\gcn_rumor\label_matrix.json', 'rb') as f_lm:
    with open(r'D:\研究生资料2020\GCN_20200211\label_matrix.json', 'rb') as f_lm:

        label_matrix = pkl.load(f_lm)

    features = np.array(features_matrix)
    features = sp.lil_matrix(features)
    adj = sp.csr_matrix(adj_matrix)
    labels = np.array(label_matrix)


    # index_list = [i for i in range(4664)]
    # random.seed(6)
    # random.shuffle(index_list)
    # train_index = index_list[:160]
    # val_index = index_list[160:3664]
    # test_index = index_list[3664:]

# train val test index
    with open(r'E:\GCN_20200211\index_save\seed2\train_index.json', 'rb') as f_ti1:
        train_index = pkl.load(f_ti1)
    with open(r'E:\GCN_20200211\index_save\seed2\val_index.json', 'rb') as f_ti2:
        val_index = pkl.load(f_ti2)
    with open(r'E:\GCN_20200211\index_save\seed2\test_index.json', 'rb') as f_ti3:
        test_index = pkl.load(f_ti3)





    random.seed(seed_train)
    train_index_list = train_index
    test_index_list = test_index
    random.shuffle(train_index_list)
    random.shuffle(test_index_list)




    train_num = 150

    # train_num = 20


    idx_train = train_index_list[0:train_num]
    idx_val = train_index_list[train_num:]
    idx_test = test_index_list

    # idx_test = val_index + test_index_list
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    # 将稀疏矩sparse_mx阵转换成tuple格式并返回
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# 处理特征:特征矩阵进行归一化并返回一个格式为(coords, values, shape)的元组
# 特征矩阵的每一行的每个元素除以行和，处理后的每一行元素之和为1
# 处理特征矩阵，跟谱图卷积的理论有关，目的是要把周围节点的特征和自身节点的特征都捕捉到，同时避免不同节点间度的不均衡带来的问题
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()

    r_inv[np.isinf(r_inv)] = 0.
    # 构建稀疏矩阵
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    return sparse_to_tuple(features)



# 邻接矩阵adj对称归一化并返回coo存储模式
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# 将邻接矩阵加上自环以后，对称归一化，并存储为COO模式，最后返回元组格式
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # 加上自环，再对称归一化
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


# 构建输入字典并返回
def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})

    # 由于邻接矩阵是稀疏的，并且用LIL格式表示，因此定义为一个tf.sparse_placeholder(tf.float32)，可以节省内存
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})

    # 49126是特征矩阵存储为coo模式后非零元素的个数（2078*1433里只有49126个非零，稀疏度达1.3%）
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

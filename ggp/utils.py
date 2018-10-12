# Author: Yin Cheng Ng
# Some data pipeline and pre-processing functions are derived from https://github.com/tkipf/gcn

import sys, os, zipfile
import scipy.sparse as sp, numpy as np, pickle as pkl, networkx as nx, tensorflow as tf
from sklearn.feature_extraction.text import TfidfTransformer
from urllib2 import urlopen, URLError, HTTPError
import GPflow
float_type = GPflow.settings.dtypes.float_type
np_float_type = np.float32 if float_type == tf.float32 else np.float64

def dlfile(url, local_file_path):
    # Open the url
    try:
        f = urlopen(url)
        print "downloading " + url
        # Open our local file for writing
        with open(local_file_path, "wb") as local_file:
            local_file.write(f.read())
    #handle errors
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url

def check_and_download_dataset(data_name):
    dataset_dir = os.path.join(os.getenv('PWD'), 'Dataset')
    if not(os.path.isdir(dataset_dir)):
        os.mkdir(dataset_dir)
    if data_name == 'citation_networks':
        data_url = 'https://www.dropbox.com/s/tln5wxqqp3o691s/citation_networks.zip?dl=1'
        data_dir = os.path.join(dataset_dir, 'citation_networks')
    else:
        raise RuntimeError('Unsupported dataset {0}'.format(data_name))
    if os.path.isdir(data_dir):
        return True
    else:
        print 'Downloading from '+data_url
        dlfile(data_url, dataset_dir+'/{0}.zip'.format(data_name))
        print 'Download complete. Extracting to '+dataset_dir
        zip_handler = zipfile.ZipFile(dataset_dir+'/{0}.zip'.format(data_name), 'r')
        zip_handler.extractall(dataset_dir)
        zip_handler.close()
        return True

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

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
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

def load_data(dataset_str, active_learning = False):
    """Load data."""
    data_path = os.getenv('PWD')+'/Dataset/citation_networks/'
    check_and_download_dataset('citation_networks')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(data_path + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(data_path + "ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    if active_learning:
        t = adj.toarray()
        sg = list(nx.connected_component_subgraphs(nx.from_numpy_matrix(t)))
        vid_largest_graph = sg[np.argmax([nx.adjacency_matrix(g).shape[0] for g in sg])].nodes()
        adj = t[vid_largest_graph,:]; adj = adj[:, vid_largest_graph]
        return sp.csr_matrix(adj), sp.csr_matrix(features.toarray()[vid_largest_graph,:]), labels[vid_largest_graph]
    else:
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data_ssl(data_name):
    adj_csr, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(data_name)
    adj_mat = np.asarray(adj_csr.toarray(), dtype=np_float_type)
    x_tr = np.reshape(np.arange(len(train_mask))[train_mask], (-1, 1))
    x_val = np.reshape(np.arange(len(val_mask))[val_mask], (-1, 1))
    x_test = np.reshape(np.arange(len(test_mask))[test_mask], (-1, 1))
    y_tr = np.asarray(y_train[train_mask], dtype=np.int32)
    y_tr = np.reshape(np.sum(np.tile(np.arange(y_tr.shape[1]), (np.sum(train_mask), 1)) * y_tr, axis=1), (-1, 1))
    y_val = np.asarray(y_val[val_mask], dtype=np.int32)
    y_val = np.reshape(np.sum(np.tile(np.arange(y_val.shape[1]), (np.sum(val_mask), 1)) * y_val, axis=1), (-1, 1))
    y_test = np.asarray(y_test[test_mask], dtype=np.int32)
    y_test = np.reshape(np.sum(np.tile(np.arange(y_test.shape[1]), (np.sum(test_mask), 1)) * y_test, axis=1), (-1, 1))
    node_features = features.toarray()
    if data_name.lower() != 'pubmed': #pubmed already comes with tf-idf
        transformer = TfidfTransformer(smooth_idf=True)
        node_features = transformer.fit_transform(node_features).toarray()
    return adj_mat, node_features, x_tr, y_tr, x_val, y_val, x_test, y_test

def load_data_al(data_name):
    adj_csr, features_csr, labels = load_data(data_name, active_learning=True)
    y = np.sum(np.tile(np.arange(labels.shape[1]), (labels.shape[0], 1)) * labels, axis=1, keepdims=True)
    y = np.asarray(y, dtype=np.int)
    x = np.reshape(np.arange(y.shape[0]), (-1,1))
    adj_mat = np.asarray(adj_csr.toarray(), dtype=np_float_type)
    node_features = features_csr.toarray()
    if data_name.lower() != 'pubmed': #pubmed already comes with tf-idf
        transformer = TfidfTransformer(smooth_idf=True)
        node_features = transformer.fit_transform(node_features).toarray()
    node_features = node_features - np.mean(node_features, axis=0)
    return adj_mat, node_features, x, y
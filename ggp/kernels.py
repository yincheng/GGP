# Author: Yin Cheng Ng

import numpy as np, tensorflow as tf
from scipy import sparse
import GPflow
from GPflow.param import DataHolder, Param
from GPflow import transforms
from ggp.utils import sparse_to_tuple
from ggp.param import SparseDataHolder

float_type = GPflow.settings.dtypes.float_type
np_float_type = np.float32 if float_type == tf.float32 else np.float64

class SparseGraphPolynomial(GPflow.kernels.Kern):
    def __init__(self, denseAdjMat, denseFeatureMat, X_tr, degree=3.0, variance=1.0, offset=1.0):
        GPflow.kernels.Kern.__init__(self, 1)
        self.degree = degree
        self.offset = Param(offset, transform=transforms.positive)
        self.variance = Param(variance, transforms.positive)
        denseAdjMat[np.diag_indices(len(denseAdjMat))] = 1.
        self.tr_features, self.tr_masks, self.tr_masks_counts = self._diag_tr_helper(denseFeatureMat, denseAdjMat, X_tr)
        self.sparse_P = SparseDataHolder(sparse_to_tuple(sparse.csr_matrix(
                        denseAdjMat/np.sum(denseAdjMat, 1, keepdims=True))))
        self.sparseFeatureMat = SparseDataHolder(sparse_to_tuple(sparse.csr_matrix(denseFeatureMat)))
        self.denseFeatureMat = DataHolder(denseFeatureMat)

    def K(self, X, X2=None):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        X2 = tf.reshape(tf.cast(X2, tf.int32), [-1]) if X2 is not None else X
        base_K_mat = (self.variance * tf.matmul(self.denseFeatureMat, self.denseFeatureMat, transpose_b = True) + self.offset) ** self.degree
        t1 = tf.sparse_tensor_dense_matmul(self.sparse_P, base_K_mat)
        t2 = tf.sparse_tensor_dense_matmul(self.sparse_P, t1, adjoint_b=True)
        return tf.gather(tf.gather(t2, X), X2, axis=1)

    def Kdiag(self, X):
        return tf.diag_part(self.K(X))

    def Kzx(self, Z, X):
        X_reshaped = tf.reshape(tf.cast(X, tf.int32), [-1])
        Kmat = (self.variance * tf.sparse_tensor_dense_matmul(self.sparseFeatureMat, Z, adjoint_b=True) + self.offset)**self.degree
        t1 = tf.sparse_tensor_dense_matmul(self.sparse_P, Kmat)
        return tf.gather(tf.transpose(t1), X_reshaped, axis=1)

    def Kzz(self, Z):
        return (self.variance * (tf.matmul(Z, Z, transpose_b=True)) + self.offset) ** self.degree

    @GPflow.param.AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def compute_Kzx(self, Z, X):
        return self.Kzx(Z)

    def Kdiag_tr(self):
        base_K_mat = self.tr_masks * (self.variance * tf.matmul(self.tr_features, self.tr_features,
                                                                 transpose_b=True) + self.offset)**self.degree
        return tf.reduce_sum(base_K_mat, [1, 2])/self.tr_masks_counts

    def _diag_tr_helper(self, node_features, adj_mat, x_tr):
        z = np.asarray([node_features[a == 1.] for a in adj_mat[x_tr.flatten()]])
        max_n = np.max([t.shape[0] for t in z])
        out = np.zeros((len(z), max_n, node_features.shape[1]))
        masks = np.zeros((len(z), max_n, max_n), dtype=np_float_type)
        for i in range(len(z)):
            out[i,:len(z[i]),:] = z[i]
            masks[i, :len(z[i]), :len(z[i])] = 1
        return DataHolder(out), DataHolder(masks), DataHolder(np.sum(np.sum(masks, 2), 1))


# Author: Yin Cheng Ng
# The code in this file is mainly derived from GPflow https://github.com/markvdw/GPflow-inter-domain/tree/master/GPflow
# with minor modifications to speed-up computation for GGP.

from __future__ import absolute_import
from GPflow.svgp import SVGP
from GPflow._settings import settings
import tensorflow as tf
float_type = settings.dtypes.float_type

class GraphSVGP(SVGP):
    # Does not support Minibatch
    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """
        # Get prior KL.
        KL = self.build_prior_KL()
        # Get conditionals
        fmean, fvar = self._elbo_helper()
        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)
        # re-scale for minibatch size
        scale = tf.cast(self.num_data, float_type)/tf.cast(tf.shape(self.X)[0], float_type)
        return tf.reduce_sum(var_exp) * scale - KL

    def _elbo_helper(self):
        # compute kernel stuff
        kern = self.kern; f = self.q_mu;
        num_data = tf.shape(self.Z)[0]  # M
        num_func = tf.shape(f)[1]  # K
        Kmn = kern.Kzx(self.Z, self.X)
        Kmm = kern.Kzz(self.Z) + tf.eye(num_data, dtype=float_type) * settings.numerics.jitter_level
        Lm = tf.cholesky(Kmm)
        # Compute the projection matrix A
        A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)
        # compute the covariance due to the conditioning
        fvar = kern.Kdiag_tr() - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([num_func, 1])
        fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N
        # another backsubstitution in the unwhitened case
        if not self.whiten:
            A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)
        # construct the conditional mean
        fmean = tf.matmul(A, f, transpose_a=True)
        if self.q_sqrt is not None:
            if self.q_sqrt.get_shape().ndims == 2:
                LTA = A * tf.expand_dims(tf.transpose(self.q_sqrt), 2)  # K x M x N
            elif self.q_sqrt.get_shape().ndims == 3:
                L = tf.matrix_band_part(tf.transpose(self.q_sqrt, (2, 0, 1)), -1, 0)  # K x M x M
                A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
                LTA = tf.matmul(L, A_tiled, transpose_a=True)  # K x M x N
            else:  # pragma: no cover
                raise ValueError("Bad dimension for q_sqrt: %s" %
                                 str(self.q_sqrt.get_shape().ndims))
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # K x N
        fvar = tf.transpose(fvar)  # N x K or N x N x K
        return fmean + self.mean_function(self.X), fvar

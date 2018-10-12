# Author: Yin Cheng Ng
# The code in this file is mainly derived from GPflow https://github.com/markvdw/GPflow-inter-domain/tree/master/GPflow

from __future__ import absolute_import
import numpy as np, tensorflow as tf
from GPflow._settings import settings
from GPflow.param import Parentable, DataHolder

float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

# when one of these attributes is set, notify a recompilation
recompile_keys = ['prior', 'transform', 'fixed']

class SparseDataHolder(DataHolder):
    """
    This behaves in the same way as DataHolder in GPflow, except that it holds a sparse array.

    """

    def __init__(self, coo_array, on_shape_change='raise'):
        """
        array is a numpy array of data.
        on_shape_change is one of ('raise', 'pass', 'recompile'), and
        determines the behaviour when the data is set to a new value with a
        different shape
        """
        Parentable.__init__(self)
        dt = self._get_type(coo_array[1])
        self._coo_array = (coo_array[0], np.asarray(coo_array[1], dtype=dt), coo_array[2])
        assert on_shape_change in ['raise', 'pass', 'recompile']
        self.on_shape_change = on_shape_change

    def _get_type(self, array):
        """
        Work out what a sensible type for the array is. if the default type
        is float32, downcast 64bit float to float32. For ints, assume int32
        """
        if any([array.dtype == np.dtype(t) for t in [np.float32, np.float64]]):
            return np_float_type
        elif any([array.dtype == np.dtype(t) for t in [np.int16, np.int32, np.int64]]):
            return np.int32
        else:
            raise NotImplementedError("unknown dtype")

    def get_feed_dict_keys(self):
        return {self: self._tf_array}

    def update_feed_dict(self, key_dict, feed_dict):
        feed_dict[key_dict[self]] = self._coo_array

    def __getstate__(self):
        d = Parentable.__getstate__(self)
        try:
            d.pop('_tf_array')
        except KeyError:
            pass
        return d

    def make_tf_array(self):
        self._tf_array = tf.sparse_placeholder(dtype=self._get_type(self._coo_array[1]),
                                               shape=[None] * len(self._coo_array[2]),
                                               name=self.name)

    def set_data(self, coo_array):
        """
        Setting a data into self._array before any TensorFlow execution.
        If the shape of the data changes, then either:
         - raise an exception
         - raise the recompilation flag.
         - do nothing
        according to the option in self.on_shape_change.
        """
        assert len(coo_array)==3, 'coo_array has to be a tuple with 3 elements (idx, 1d numpy array, shape)'
        if self.shape == coo_array[2]:
            self._coo_array = coo_array  # just accept the new values
        else:
            if self.on_shape_change == 'raise':
                raise ValueError("The shape of this data must not change. \
                                  (perhaps make the model again from scratch?)")
            elif self.on_shape_change == 'recompile':
                self._coo_array = coo_array.copy()
                self.highest_parent._needs_recompile = True
            elif self.on_shape_change == 'pass':
                self._coo_array = coo_array.copy()
            else:
                raise ValueError('invalid option')  # pragma: no cover

    @property
    def value(self):
        return self._coo_array.copy()

    @property
    def size(self):
        return np.multiply(*self._coo_array[2])

    @property
    def shape(self):
        return self._coo_array[2]

    def __str__(self, prepend='SparseData:'):
        return prepend + \
               '\033[1m' + self.name + '\033[0m' + \
               '\n' + str(self.value)

# Author: Yin Cheng Ng

from ggp.utils import *
from ggp.kernels import SparseGraphPolynomial
from ggp.model import GraphSVGP
from scipy.cluster.vq import kmeans2
import numpy as np
import os, time, pickle, argparse

class SSLExperiment(object):
    def __init__(self, data_name, random_seed):
        self.data_name = data_name.lower()
        self.random_seed = int(random_seed); np.random.seed(self.random_seed); tf.set_random_seed(self.random_seed)
        # Load data
        self.adj_mat, self.node_features, self.x_tr, self.y_tr, self.x_val, self.y_val, self.x_test, self.y_test \
            = load_data_ssl(self.data_name)
        # Init kernel
        k = SparseGraphPolynomial(self.adj_mat, self.node_features, self.x_tr, degree=3.)
        k.offset = np.abs(np.random.randn(1) + 5.); k.offset.fixed = False
        k.variance = 1.; k.variance.fixed = True
        # Init inducing points
        ind_points = kmeans2(self.node_features, len(self.x_tr), minit='points')[0]
        # Init optimizer
        self.optimizer = tf.train.AdamOptimizer(0.0005)
        # Init model
        self.m = GraphSVGP(self.x_tr, self.y_tr, k, GPflow.likelihoods.MultiClass(len(np.unique(self.y_tr))), ind_points,
                      num_latent=len(np.unique(self.y_tr)), minibatch_size=len(self.x_tr), whiten=True, q_diag=False)
        # Define housekeeping variables
        self.last_ts = time.time()
        self.iter = 0; self.check_obj_every = 200
        self.log_iter = []; self.log_t = []; self.log_obj = []; self.log_param = None; self.log_opt_state = None;
        self.param_fp = os.path.join(os.getenv('PWD'), 'ssl_param_files')
        if not (os.path.isdir(self.param_fp)):
            os.mkdir(self.param_fp)
        self.param_fp = os.path.join(self.param_fp, 'SSL-{0}-rs_{1}.p'.format(self.data_name, random_seed))
        self.m._compile(self.optimizer)
        if os.path.isfile(self.param_fp):
            print 'Param. file already exists! Loading from {0}.'.format(self.param_fp)
            self.load_snapshot(self.param_fp)
        else:
            self.save_snapshot(self.param_fp, update_before_saving=True)

    def update_log(self, param):
        self.log_t.append(time.time() - self.last_ts)
        self.log_param = param.copy()
        self.log_opt_state = self.m.get_optimizer_variables()[0]
        self.log_obj.append(self.m._objective(param)[0])
        self.log_iter.append(self.iter);
        self.m.set_state(param)

    def save_snapshot(self, pickle_fp, update_before_saving = False):
        log_up_to_date = self.iter >= self.log_iter[-1] if len(self.log_iter)>1 else False
        if update_before_saving and not(log_up_to_date):
            param = self.m.get_free_state()
            self.update_log(param)
        p_dict = {}
        p_dict['iter'] = self.iter; p_dict['log_iter'] = self.log_iter; p_dict['log_t'] = self.log_t;
        p_dict['log_obj'] = self.log_obj;
        p_dict['log_opt_state'] = self.log_opt_state; p_dict['log_param'] = self.log_param;
        pickle.dump(p_dict, open(pickle_fp, "wb"))

    def load_snapshot(self, pickle_fp):
        p_dict = pickle.load(open(pickle_fp, 'rb'))
        self.iter = p_dict['iter'];
        self.log_iter = p_dict['log_iter'];
        self.log_t = p_dict['log_t'];
        self.log_param = p_dict['log_param'];
        self.log_opt_state = p_dict['log_opt_state']
        self.log_obj = p_dict['log_obj'];
        self.m.set_optimizer_variables_value(self.log_opt_state.copy())
        self.m.set_state(self.log_param.copy())

    def _callback(self, param):
        self.iter += 1
        if (self.iter % self.check_obj_every) == 0:
            self.update_log(param)
            self.last_ts = time.time()
            print 'SSL-{0}-rs_{1}'.format(self.data_name, self.random_seed), self.log_iter[-1], self.log_obj[-1], \
                 '({0:.3f}s)'.format(self.log_t[-1])
            # Save a snapshot of the model with the best ELBO
            if self.log_obj[-1] < np.min(np.array(self.log_obj)[:-1]):
                self.save_snapshot(self.param_fp)

    def train(self, maxiter, check_obj_every_n_iter = None):
        self.check_obj_every = self.check_obj_every if check_obj_every_n_iter is None else check_obj_every_n_iter
        if self.iter < maxiter:
            self.last_ts = time.time()
            self.m.optimize(method=self.optimizer, maxiter=maxiter - self.iter, callback=self._callback)
        print '{0} iterations completed.'.format(self.iter)

    def evaluate(self):
        print '\nEvaluating prediction accuracies...'
        # Restore the parameters to the one with the best ELBO
        tmp_params = self.m.get_free_state().copy()
        self.m.set_state(pickle.load(open(self.param_fp, 'rb'))['log_param'])
        tr_acc = np.sum(np.argmax(self.m.predict_y(self.x_tr)[0], 1) == self.y_tr.flatten())*100./len(self.y_tr)
        test_acc = np.sum(np.argmax(self.m.predict_y(self.x_test)[0], 1) == self.y_test.flatten())*100./len(self.y_test)
        print 'Prediction Accuracies: '
        print '\tTraining Set: {0:.2f}%'.format(tr_acc)
        print '\tTest Set: {0:.2f}%'.format(test_acc)
        # Revert the parameters to the original values
        self.m.set_state(tmp_params)
        return {'train': tr_acc, 'test': test_acc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="data set name [cora|citeseer|pubmed]", type=str)
    parser.add_argument("rs", help="random seed [integer]", type=int)
    parser = parser.parse_args()
    exp_obj = SSLExperiment(parser.data, parser.rs)
    exp_obj.train(10000, check_obj_every_n_iter=200)
    exp_obj.evaluate()

if __name__ == "__main__":
    main()
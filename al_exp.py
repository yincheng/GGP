# Author: Yin Cheng Ng

from ggp.utils import *
from ggp.kernels import SparseGraphPolynomial
from ggp.model import GraphSVGP
from scipy.cluster.vq import kmeans2
import numpy as np
import os, time, pickle, argparse

class ALExperiment(object):
    def __init__(self, data_name, random_seed):
        self.data_name = data_name.lower()
        self.random_seed = int(random_seed); np.random.seed(self.random_seed); tf.set_random_seed(self.random_seed)
        # Load data
        self.adj_mat, self.node_features, self.all_x, self.all_y = load_data_al(self.data_name)
        # Define housekeeping variables
        self.tr_mask = np.array([False]*self.all_x.shape[0]); self.tr_mask[random_seed-1] = True;
        self.res_list = [{'vid':random_seed-1, 'test_acc':np.NaN, 't_delta':np.NaN}]
        self.n_sample_acquired = 1;
        self.result_fp = os.path.join(os.getenv('PWD'), 'al_result_files')
        if not (os.path.isdir(self.result_fp)):
            os.mkdir(self.result_fp)
        self.result_fp = os.path.join(self.result_fp, 'AL-{0}-rs_{1}.p'.format(self.data_name, random_seed))

    def learn(self, n_sample_budget, maxitr_opt):
        while self.n_sample_acquired < n_sample_budget:
            self.res_list.append({'vid':-100, 'test_acc':np.NaN, 't_delta':np.NaN})
            self.res_list[-2]['test_acc'], self.res_list[-1]['vid'], self.res_list[-2]['t_delta'] \
                = self.eval_and_acq(maxitr_opt)
            assert np.sum(self.tr_mask) == self.n_sample_acquired, 'Num. of sample in tr_mask != n_sample_acquired'
            assert self.tr_mask[self.res_list[-1]['vid']]==False, 'Node {0} alrdy acq.'.format(self.res_list[-1]['vid'])
            self.tr_mask[self.res_list[-1]['vid']] = True; self.n_sample_acquired += 1
            self.print_res_list()
        self.res_list[-1]['test_acc'], _, self.res_list[-1]['t_delta']= self.eval_and_acq(maxitr_opt);
        self.print_res_list()
        print '\nSaving result to {0}'.format(self.result_fp)
        pickle.dump(self.res_list, open(self.result_fp, 'wb'))

    def eval_and_acq(self, maxitr):
        ts = time.time()
        # Setup and train model
        m, opt = self.setup_model_and_opt(); m.optimize(method=opt, maxiter=maxitr)
        # Predict and evaluate test accuracy
        pred_lab = np.argmax(m.predict_y(self.all_x[~self.tr_mask])[0], 1).flatten()
        target_lab = self.all_y[~self.tr_mask].flatten()
        test_acc = np.sum(pred_lab == target_lab)*100./len(target_lab)
        # SOPT acquisition function
        laplacian = np.diag(np.sum(self.adj_mat, 1)) - self.adj_mat
        predCovCQ = np.zeros((len(laplacian), len(laplacian)))
        masks = np.reshape(~self.tr_mask, (-1,1)) & np.reshape(~self.tr_mask, (1,-1))
        predCovCQ[masks] = np.linalg.inv(laplacian[~self.tr_mask][:,~self.tr_mask]).flatten()
        acq_scores = np.sum(predCovCQ[~self.tr_mask][:, ~self.tr_mask], 1)/np.sqrt(np.diag(predCovCQ)[~self.tr_mask])
        to_label = self.all_x[~self.tr_mask][np.argmax(acq_scores)]
        # Delete the model and optimizer in preparation for the next iteration
        del m; del opt
        return test_acc, to_label[0], time.time()-ts

    def setup_model_and_opt(self):
        n_class = len(np.unique(self.all_y))
        x_tr = self.all_x[self.tr_mask]; y_tr = self.all_y[self.tr_mask]; n_inducing_points = len(x_tr)
        k = SparseGraphPolynomial(self.adj_mat, self.node_features, x_tr, degree=1.)
        k.offset = 0.; k.offset.fixed = True; k.variance = 1.; k.variance.fixed = True
        ind_points = kmeans2(self.node_features, n_inducing_points, minit='points')[0]
        opt = tf.train.AdamOptimizer(0.005)
        m = GraphSVGP(x_tr, y_tr, k, GPflow.likelihoods.MultiClass(n_class), ind_points,
                      num_latent=n_class, minibatch_size=len(x_tr), whiten=True, q_diag=False)
        return m, opt

    def print_res_list(self):
        print ' '
        print 'n_sample_acquired', 'test_acc', 'next_sample_id(true label)', 'cpu_time'
        for i, d in enumerate(self.res_list):
            print i+1, '{0:.2f}'.format(d['test_acc']), str(d['vid'])+'({0})'.format(self.all_y[d['vid']][0]), \
                '{0:.2f}s'.format(d['t_delta'])
        print 'AL-{0}-{1}'.format(self.data_name, self.random_seed)
        print ' '

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="data set [cora|citeseer]", type=str)
    parser.add_argument("rs", help="random seed [integer]", type=int)
    parser = parser.parse_args()
    exp_obj = ALExperiment(parser.data, parser.rs)
    exp_obj.learn(50, 300)

if __name__ == "__main__":
    main()

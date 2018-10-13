# Graph Gaussian Process (GGP)
The code and data in this repository accompany the paper `[Bayesian Semi-supervised Learning with Graph Gaussian Processes](https://arxiv.org/abs/1809.04379)'
```
@inproceedings{ng2018gaussian,
  title={Bayesian semi-supervised learning with graph Gaussian processes},
  author={Ng, Yin Cheng and Colombo, Nicolo and Silva, Ricardo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
```

The code depends on a branch of GPflow located [here](https://github.com/markvdw/GPflow-inter-domain).

To run the graph-based semi-supervised learning experiment, execute the following command:
 ```
 python ssl_exp.py [name of the data set] [random seed]
 valid options for the name of the data set are: cora, citeseer or pubmed
 valid options for the random seed: any integer
 ```

To run the active learning experiment, execute the following command:
 ```
 python al_exp.py [name of the data set] [random seed]
 valid options for the name of the data set are: cora or citeseer
 valid options for the random seed: any integer
 ```

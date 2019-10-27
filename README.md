# Collaborative Preference Embedding against Sparse labels
> Shilong Bao, Qianqian Xu, Ke Ma, Zhiyong Yang, Xiaochun Cao and Qingming Huang. ACM Conference on Multimedia (ACM MM), 2019. 

We have implemented our model using Tensorflow and we run our code on Ubuntu 18.04 system with CPU, since AdaGrad does not seem to work on GPU.



# Abstract
 In the paper, we proposed a novel method named as **Collaborative Preference Embedding**(CPE) which can directly deal with sparse and insufficient user preference information. Specifically, we designed two schemes specifically against the limited generalization ability in terms of sparse labels.
# Methodology
## Framework
  ![img](https://github.com/statusrank/CPE/blob/master/img/framework.png)
## Why we optimize the margin distribution.
  ![img](https://github.com/statusrank/CPE/blob/master/img/figure1.png)

# Experiment
  We conduct comprehensive experiments to demonstrate the superiority of CPE. Empirical results on three different benchmark datasets, including MovieLens-100K, CiteULike-T and BookCrossing, consistently show that our method can achieve reasonable generalization performance even when suffering sparse preference information.

# Acknowledgment 
This implementation is based on [CML](https://github.com/changun/CollMetric). We sincerely thank the contributions of the authors.

# Requirements
  - python >= 3.5
  - Tensorflow
  - tqdm
  - scipy
  - numpy
  - scikit-learn
  - functools
  - toolz

# Citation
Please cite our paper if you use this code in your own work.
> @inproceedings{bao2019,
 title={Collaborative Preference Embedding against Sparse Labels},
 author={Bao, Shilong and Xu, qianqian and Ma, Ke and Huang, Qingming and Cao, Xiaochun},
 booktitle={2019 ACM Conference on Multimedia},
 year={2019}
}

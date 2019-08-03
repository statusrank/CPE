# Collaborative Preference Embedding against Sparse labels
> Shilong Bao, Qianqian Xu, Ke Ma, Zhiyong Yang, Xiaochun Cao and Qingming Huang. ACM Conference on Multimedia (ACM MM), 2019. 

We have implemented our model using Tensorflow and we run our code on Ubuntu 18.04 system.


# Abstract
 In the paper, we proposed a novel method named as **Collaborative Preference Embedding**(CPE) which can directly deal with sparse and insufficient user preference information. Specifically, we designed two schemes specifically against the limited generalization ability in terms of sparse labels.
# Methodology
## Framework
  ![img](https://github.com/statusrank/CPE/blob/master/img/framework.png)
## Why we optimize the margin distribution.
  ![img](https://github.com/statusrank/CPE/blob/master/img/figure1.png)(./pic/pic1_50.png =100x100)

# Experiment
  We conduct comprehensive experiments to demonstrate the superiority of CPE. Empirical results on three different benchmark datasets, including MovieLens-100K, CiteULike-T and BookCrossing, consistently show that our method can achieve reasonable generalization performance even when suffering sparse preference information.

# Acknowledgment 
This implementation is based on [CML](https://github.com/changun/CollMetric). We sincerely thank the contributions of the authors.

# Requirements
  - python3 
  - Tensorflow
  - tqdm
  - scipy
  - numpy
  - scikit-learn

# Citation
Please cite our paper if you use this code in your own work.

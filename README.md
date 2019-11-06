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
```
@inproceedings{DBLP:conf/mm/BaoXMYCH19,
  author    = {Shilong Bao and
               Qianqian Xu and
               Ke Ma and
               Zhiyong Yang and
               Xiaochun Cao and
               Qingming Huang},
  title     = {Collaborative Preference Embedding against Sparse Labels},
  booktitle = {Proceedings of the 27th {ACM} International Conference on Multimedia,
               {MM} 2019, Nice, France, October 21-25, 2019},
  pages     = {2079--2087},
  year      = {2019}
}
```

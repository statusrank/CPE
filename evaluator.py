"""
evaluator.py
"""
import tensorflow as tf
import math,random
from scipy.sparse import lil_matrix
import numpy as np

class RecallEvaluator(object):
    def __init__(self, model, train_user_item_matrix = None, test_user_item_matrix = None):
        self.model = model
        self.n_users , self.n_items = test_user_item_matrix.shape
        if train_user_item_matrix is not None:
            self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        else:
            self.train_user_item_matrix = None
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        #n_users , n_items = test_user_item_matrix.shape
        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                    for u in range(self.n_users) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                        for u in range(self.n_users) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0

    def precision_recall_ndcg_k(self, sess, users, k=30):

        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        _, user_tops = sess.run(tf.nn.top_k(self.model.user_item_scores, k + self.max_train_count),
                                {self.model.user_predicted_ids: users})
        recall_k,precision_k,NDCG_k = [],[],[]
        for user_id, tops in zip(users, user_tops):
            if self.train_user_item_matrix is not None:
                train_set = self.user_to_train_set.get(user_id, set())
            else:
                train_set = set()
            test_set = self.user_to_test_set.get(user_id, set())

            n_k = k if len(test_set) > k else len(test_set)
            if len(test_set) >= k:
                _idcg,_dcg = 0,0
                for pos in range(n_k):
                    _idcg += 1.0 / math.log(pos + 2,2)
                new_set = []
                top_k = 0
                for val in tops:
                    if val in train_set:
                        continue
                    top_k += 1
                    new_set.append(val)
                    if top_k >= k:
                        break
                hits = [(idx,val) for idx,val in enumerate(new_set) if val in test_set]
                cnt = len(hits)
                for idx in range(cnt):
                    _dcg += 1.0 /math.log(hits[idx][0] + 2,2)
                precision_k.append(float(cnt / k))
                recall_k.append(float(cnt / len(test_set)))
                NDCG_k.append(float(_dcg / _idcg))
        return precision_k,recall_k,NDCG_k
    def map_mrr_auc_ndcg(self,sess,users):
        _, user_tops = sess.run(tf.nn.top_k(self.model.user_item_scores, self.n_items),
                                {self.model.user_predicted_ids: users})
        
        MAP,MRR,auc,NDCG = [],[],[],[]
        for user_id, tops in zip(users, user_tops):
            if self.train_user_item_matrix is not None:
                train_set = self.user_to_train_set.get(user_id, set())
            else:
                train_set = set()
            test_set = self.user_to_test_set.get(user_id, set())
            if len(test_set) >= 30:
                new_set = []
                for val in tops:
                    if val in train_set:
                        continue
                    new_set.append(val)
                _idcg = 0
                _dcg = 0
                for i in range(len(test_set)):
                    _idcg += 1/ math.log(i + 2,2)
                _ap = 0
                hits = [(idx,val) for idx,val in enumerate(new_set) if val in test_set]
                cnt = len(hits)
                for c in range(cnt):
                    _ap += float((c + 1) / (hits[c][0] + 1))
                    _dcg += 1 / math.log(hits[c][0] + 2,2)
                if cnt != 0:
                    MAP.append(float(_ap / cnt))
                    MRR.append(float(1 / (hits[0][0] + 1)))
                    NDCG.append(float(_dcg / _idcg))
                labels = [1 if item in test_set else 0 for item in new_set]
                auc.append(self.AUC(labels,len(test_set)))
        return MAP,MRR,auc,NDCG
    def AUC(self,labels,_K):
        if len(labels) <= _K:
            return 1
        auc = 0
        for i, label in enumerate(labels[::-1]):
            auc += label * (i + 1)

        return (auc - _K * (_K + 1) / 2) / (_K * (len(labels) - _K))

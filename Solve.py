import tensorflow as tf
import numpy as np
from evaluator import RecallEvaluator
from Sampler import Sampler
from utils import read_data,split_data
from tqdm import tqdm
import functools,toolz
from scipy.sparse import dok_matrix,lil_matrix
import os

def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs,reuse=tf.AUTO_REUSE):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class CPE(object):
    def __init__(self,
                num_users,
                num_items,
                embed_dim = 20,
                gamma_0 = 1.0,
                learning_rate = 0.1,
                use_cov_loss = True,
                cov_loss_reg = 0.1,
                clip_norm = 1.0):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.gamma_0 = gamma_0
        self.learning_rate = learning_rate
        self.use_cov_loss = use_cov_loss
        self.cov_loss_reg = cov_loss_reg
        self.clip_norm = clip_norm
        
        ## user-item pairs,sample negative items,user predict 
        self.user_item_pos_pairs = tf.placeholder(tf.int32,[None,2],name = "user_item_pos_pairs")
        self.neg_items = tf.placeholder(tf.int32,[None,None],name = "neg_items")
        self.user_predicted_ids = tf.placeholder(tf.int32,[None],name = "user_predicted_ids")

        self.users_embedding
        self.items_embedding
        self.Embedding_loss
        self.loss
        self.optimize
    @define_scope
    def users_embedding(self):
        return tf.Variable(tf.random_uniform([self.num_users,self.embed_dim],
                                            minval = -1,
                                            maxval = 1,
                                            dtype = tf.float64))
    @define_scope
    def items_embedding(self):
        return tf.Variable(tf.random_uniform([self.num_items,self.embed_dim],
                                            minval = -1,
                                            maxval = 1,
                                            dtype = tf.float64))
        
    @define_scope
    def Embedding_loss(self):
        users = tf.nn.embedding_lookup(self.users_embedding,
                                        self.user_item_pos_pairs[:,0],
                                        name = "users")
            
        pos_items = tf.nn.embedding_lookup(self.items_embedding,
                                            self.user_item_pos_pairs[:,1],
                                            name = "pos_items")

        pos_distance = tf.reduce_sum(tf.squared_difference(users,pos_items),1,name = "pos_distance")
            
        

        neg_items = tf.transpose(tf.nn.embedding_lookup(self.items_embedding,self.neg_items),
                                (0,2,1),name = "neg_items")
        

        neg_distance = tf.reduce_sum(tf.squared_difference(tf.expand_dims(users,-1),neg_items),
                                    1,name = "neg_distance")

        gamma_s = neg_distance - tf.expand_dims(pos_distance,-1)
        loss = tf.reduce_sum(tf.maximum(gamma_s - self.gamma_0,0) + tf.maximum(self.gamma_0 - gamma_s,0))

        return loss
        
    @define_scope
    def Cov_loss(self):
        if self.use_cov_loss:
            X = tf.concat((self.items_embedding,self.users_embedding),0)
            number = tf.cast(tf.shape(X)[0],tf.float64)
            X = X - (tf.reduce_mean(X,axis = 0))
            C = tf.matmul(X,X,transpose_a = True) / number
            eig = tf.linalg.eigvalsh(C,name = "eig")
            log_determinant_of_C = tf.reduce_sum(tf.log(eig))
            return (tf.trace(C) - log_determinant_of_C) * self.cov_loss_reg
    @define_scope
    def loss(self):

        loss = self.Embedding_loss
        if self.use_cov_loss:
            loss += self.Cov_loss
        return loss
        
    @define_scope
    def clip_by_norm_op(self):

        return [tf.assign(self.users_embedding,tf.clip_by_norm(self.users_embedding,
                                                                self.clip_norm,axes = [1])),
                tf.assign(self.items_embedding,tf.clip_by_norm(self.items_embedding,
                                                                self.clip_norm,axes = [1]))]
        
    @define_scope
    def optimize(self):
        grads = []
        grads.append(tf.train.AdagradOptimizer(self.learning_rate)
                    .minimize(self.loss,var_list = [self.users_embedding,self.items_embedding]))
        
        with tf.control_dependencies(grads):
            return grads + [self.clip_by_norm_op]
        
    @define_scope
    def user_item_scores(self):

        users = tf.expand_dims(tf.nn.embedding_lookup(self.users_embedding,self.user_predicted_ids),1)
            
        items = tf.expand_dims(self.items_embedding,0)
            
        return tf.cast(-tf.reduce_sum(tf.squared_difference(users, items), 2, name="scores"),
                        dtype = tf.float64)
        
BATCH_SIZE = 16
NUMBER_NEGATIVE_ITEMS = 30
EVALUATION_EVERY_N_BATCHES = 10
EMBED_DIM = 300
VALID_USERS_NUMBERS = 1000

model_saving_path = "MovieModel"
model_name = "Solver"
data_path_base = "ml-100k/"
def optimize(model,sampler,train,valid,total_batch):
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    saver.save(sess,os.path.join(model_saving_path,model_name))
    # Sample users to calculate recall validation
    validation_users = set(list(set(valid.nonzero()[0])))
    #validation_users = np.random.choice(list(set(valid.nonzero()[0])),size = len(valid_users),replace = False)
    # early-stopping
    k1 ,k2 = 30,50
    epoch = 0
    Recall = RecallEvaluator(model = model,train_user_item_matrix = train, test_user_item_matrix = valid)

    while True:
        
        # Trian model
        Loss = []
        for i in tqdm(range(total_batch),desc = "Training..."):
            user_item_pairs,neg_item = sampler.next_batch()
            _,loss = sess.run((model.optimize,model.loss),{model.user_item_pos_pairs: user_item_pairs,
                                                            model.neg_items: neg_item})
            Loss.append(loss)
        print("epoch :{} loss : {}".format(epoch,np.mean(Loss)))
        epoch += 1
        recalls_k1,precisions_k1,ndcgs_k1 = [],[],[]
        recalls_k2,precisions_k2,ndcgs_k2 = [],[],[]
        _maps,_mrrs,_aucs  = [],[],[]
        #validation_users = np.random.choice(list(set(valid.nonzero()[0])),size = VALID_USERS_NUMBERS,
        #                                    replace = False)
        for users_chunk in toolz.partition_all(100,validation_users):
            precision_k1,recall_k1,ndcg_k1 =Recall.precision_recall_ndcg_k(
                                                sess = sess,users = users_chunk,k = k1)
            precisions_k1.extend(precision_k1)
            recalls_k1.extend(recall_k1)
            ndcgs_k1.extend(ndcg_k1)

            precision_k2,recall_k2,ndcg_k2 = Recall.precision_recall_ndcg_k(
                                                sess = sess,users = users_chunk,k = k2)
            precisions_k2.extend(precision_k2)
            recalls_k2.extend(recall_k2)
            ndcgs_k2.extend(ndcg_k2)

            _map,_mrr,_auc,_  = Recall.map_mrr_auc_ndcg(sess = sess,users = users_chunk)
            _maps.extend(_map)
            _mrrs.extend(_mrr)
            _aucs.extend(_auc)
        print("+"*20)
        print("P@" + str(k1) + ": {}".format(np.mean(precisions_k1)))    
        print("R@" + str(k1) + ": {}".format(np.mean(recalls_k1)))
        print("NDCG@" + str(k1) + ": {}".format(np.mean(ndcgs_k1)))
        print("-"*20)
        print("P@" + str(k2) + ": {}".format(np.mean(precisions_k2)))    
        print("R@" + str(k2) + ": {}".format(np.mean(recalls_k2)))
        print("NDCG@" + str(k2) + ": {}".format(np.mean(ndcgs_k2)))
        print("-"*20)
        print("MAP: {}".format(np.mean(_maps)))
        print("MRR: {}".format(np.mean(_mrrs)))
        print("AUC: {}".format(np.mean(_aucs)))
        print("+"*20)
        saver.save(sess,os.path.join(model_saving_path,model_name))
    sess.close()

def Savedata(test_matrix,num_users,num_items,base_path,data_path):
    test_matrix = lil_matrix(test_matrix)
    with open(base_path + data_path,"w") as f:
        f.write(str(num_users) + " " + str(num_items) + "\n")
        for user in tqdm(range(num_users),desc = "Save the test/train data"):
            items = list(test_matrix.rows[user])
            num_item = len(items)
            f.write(str(num_item) + " ")
            for item in items:
                f.write(str(item) + " ")
            f.write("\n")
if __name__ == '__main__':
    
    user_item_matrix = read_data(user_path =  data_path_base + "users.dat")
    num_users,num_items = user_item_matrix.shape
    print("Number of users:{} and items:{}".format(num_users,num_items))
    train_matrix,valid_matrix,test_matrix = split_data(user_item_matrix = user_item_matrix,
                                                        split_ratio = (4,1,0))
    # Save the test data.
    #Savedata(test_matrix,num_users,num_items,base_path = data_path_base,data_path = 'test.dat')
    #Save the train data.
    #Savedata(train_matrix,num_users,num_items,base_path = data_path_base,data_path = 'train.dat')

    sampler = Sampler(  user_item_matrix = train_matrix,
                        batch_size = BATCH_SIZE,
                        num_negative_samples = NUMBER_NEGATIVE_ITEMS,
                        n_workers = 3,
                        check_neg = True)

    model = CPE(num_users = num_users,
                num_items = num_items,
                embed_dim = EMBED_DIM,
                gamma_0 = 1.25,
                learning_rate = 0.05,
                use_cov_loss = True,
                cov_loss_reg = 0.005,
                clip_norm = 1.1)
    optimize(model,sampler,train_matrix,valid_matrix,int(len(train_matrix.nonzero()[0]) / BATCH_SIZE))

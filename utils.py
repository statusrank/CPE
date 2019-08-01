"""
utils.py 
user(row)-item(column)
"""
import numpy as np
from collections import defaultdict
from scipy.sparse import dok_matrix,lil_matrix
from tqdm import tqdm

def read_data(user_path):
    # defaultdict
    user_dict = defaultdict(set) # set to remove duplicate
    # readlines is faster than readline
    # read user-item data
    num_users = 0
    for u,u_liked_items in enumerate(open(user_path).readlines()):
        items = u_liked_items.strip().split()
        # the first number is ignored ,because is the number of items the user liked.
        num_users += 1
        if int(items[0]) == 0:
            continue
        for item in items[1:]:
            user_dict[u].add(int(item))
    #num_users = len(user_dict)
    #print(num_users)
    num_items = max([item for items in user_dict.values() for item in items]) + 1
    
    user_item_matrix = dok_matrix((num_users,num_items),dtype = np.int32)
    for u,u_liked_items in enumerate(open(user_path).readlines()):
        items = u_liked_items.strip().split()
        for item in items[1:]:
            user_item_matrix[u,int(item)] = 1 # sparse matrix
    return user_item_matrix
def split_data(user_item_matrix,split_ratio = (3,1,1),seed = 1):
    #set the seed to have deterministic results
    np.random.seed(seed)
    train_matrix = dok_matrix(user_item_matrix.shape)
    val_matirx = dok_matrix(user_item_matrix.shape)
    test_matrix = dok_matrix(user_item_matrix.shape)
    
    # convert it to lili format to process it fast.
    user_item_matrix = lil_matrix(user_item_matrix)
    num_users = user_item_matrix.shape[0]
    for user in tqdm(range(num_users),desc = "Split data into train/valid/test"):
        items = list(user_item_matrix.rows[user])
        if len(items) >= 5:
            np.random.shuffle(items)
        # slice index must be integer
            train_count = int(len(items) * split_ratio[0] / sum(split_ratio))
            valid_count = int(len(items) * split_ratio[1] / sum(split_ratio))
            for i in items[0:train_count]:
                train_matrix[user,i] = 1
            for i in items[train_count:train_count + valid_count]:
                val_matirx[user,i] = 1
            for i in items[train_count + valid_count:]:
                test_matrix[user,i] = 1
    print("split the data into trian/validatin/test {}/{}/{} ".format(
        len(train_matrix.nonzero()[0]),
        len(val_matirx.nonzero()[0]),
        len(test_matrix.nonzero()[0])))
    return train_matrix,val_matirx,test_matrix
def read_test_data(data_path):
    user_dict = defaultdict(set)
    num_users,num_items = open(data_path).readline().split()
    user_item_matrix = dok_matrix((int(num_users),int(num_items)),dtype = np.int32)
    line = 0
    for u,u_liked_items in enumerate(open(data_path).readlines()):
        if line == 0:
            line += 1
            continue
        items = u_liked_items.strip().split()
        if int(items[0]) == 0:
            continue
        for item in items[1:]:
            user_item_matrix[u,int(item)] = 1
    return user_item_matrix

def read_train_data(data_path):
    user_dict = defaultdict(set)
    num_users,num_items = open(data_path).readline().split()
    #print(num_users,num_items)
    user_item_matrix = dok_matrix((int(num_users),int(num_items)),dtype = np.int32)
    line = 0
    for u,u_liked_items in enumerate(open(data_path).readlines()):
        if line == 0:
            line += 1
            continue
        items = u_liked_items.strip().split()
        if int(items[0]) == 0:
            continue
        for item in items[1:]:
            user_item_matrix[u,int(item)] = 1
    return user_item_matrix
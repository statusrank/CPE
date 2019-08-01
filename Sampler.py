import numpy as np
from scipy.sparse import lil_matrix
from multiprocessing import Process,Queue

def sample_fun(user_item_matrix,batch_size,num_negative_sample,result_queue,check_neg = True):
    user_item_matrix = lil_matrix(user_item_matrix)
    # Convert matrix to positive  user-item pairs
    user_item_pos_pairs = np.asarray(user_item_matrix.nonzero()).T
    user_pos_set = {u: set(row) for u,row in enumerate(user_item_matrix.rows)}
    while True:
        np.random.shuffle(user_item_pos_pairs)
        for i in range(int(len(user_item_pos_pairs) / batch_size)):
            pos_pairs = user_item_pos_pairs[i * batch_size:(i + 1) * batch_size,:]
            neg_items = np.random.randint(0,user_item_matrix.shape[1],
                                            size = (len(pos_pairs),num_negative_sample))
            ## Check if we sample any positive items as negative samples.
            if check_neg:
                for pos_pair,neg_item,i in zip(pos_pairs,neg_items,
                                                range(len(neg_items))):
                    user = pos_pair[0]
                    for j ,neg in enumerate(neg_item):
                        while neg in user_pos_set[user]:
                            neg_items[i,j] = neg = np.random.randint(0,user_item_matrix.shape[1])
            
            result_queue.put((pos_pairs,neg_items))

class Sampler(object):
    def __init__(self,user_item_matrix,batch_size = 10000,num_negative_samples = 10,
                n_workers = 5,check_neg = True):
        self.result_queue = Queue(maxsize = n_workers * 2)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target = sample_fun,args = (user_item_matrix,
                                                    batch_size,
                                                    num_negative_samples,
                                                    self.result_queue,
                                                    check_neg)))
            self.processors[-1].start()
    
    def next_batch(self):
        return self.result_queue.get()
    
    def close(self):
        for process in self.processors: # Process 
            process.terminate()
            process.join()        
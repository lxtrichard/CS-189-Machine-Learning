# TODO: put the path to your 'hw6_mds189', which should contain a 'trainval' and 'test' directory
path = '../mds189/trainval'
import numpy as np
from data_utils import load_mds189
# load the dataset
debug = False  # OPTIONAL: you can change this to True for debugging *only*. Your reported results must be with debug = False
feat_train, label_train, feat_val, label_val = load_mds189(path,debug)
from solver import Solver

from classifiers.fc_net import FullyConnectedNet

import itertools

data = {
      'X_train': feat_train,
      'y_train': label_train,
      'X_val': feat_val,
      'y_val': label_val}

best_val = 0.0
best_model = None
best_combo = None

def train_model(combo):
    hd,lrd,ne,bs,lre,ws = combo
    model = FullyConnectedNet(input_dim=75,
                              hidden_dim=hd,
                              weight_scale=ws)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': lre,
                    },
                    lr_decay=lrd,
                    num_epochs=ne, 
                    batch_size=bs,
                    verbose=False
                    #print_every=100
                   )
    solver.train()
    return solver
    
lr_decay = [0.95,0.99,1.0]
num_epochs = [10,15]
batch_size = [64,128,256]
learning_rate = [1e-4,1e-3,1e-2]
weight_scale = [1e-4,1e-3,1e-2]
hidden_dims = [[100]*i for i in range(1,5)]

prog = 1
combos = [combo for combo in itertools.product\
(hidden_dims,lr_decay,num_epochs,batch_size,learning_rate,weight_scale)]
total = str(len(combos))

for combo in combos:
    model = train_model(combo)
    avg_val = np.mean(model.val_acc_history)
    if avg_val > best_val:
        best_val = avg_val
        best_model = model
        best_combo = combo
    print("\rfinished training model # "+str(prog)+"/"+total,end="")
    prog += 1
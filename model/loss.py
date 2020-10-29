import numpy as np

# loss            #output , y
def cross_entropy(inputs, labels):

    out_num = labels.shape
    #p = np.sum(labels.reshape(1,out_num)*inputs)
    p=abs(labels-inputs)
    loss = -np.log(p)
    return loss



import matplotlib.pyplot as plt



# Overlapping plots
def train_val_graph(epoch_n, train_loss_list, val_loss_list):
    plt.plot(range(epoch_n),train_loss_list,'b',label='Train Loss')
    plt.plot(range(epoch_n),val_loss_list,'g',label='Val Loss')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend()



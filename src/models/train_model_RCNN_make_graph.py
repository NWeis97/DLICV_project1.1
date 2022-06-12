# Import 
import os
import numpy as np
import PIL.Image as Image

import matplotlib.pyplot as plt
import pdb
import pandas as pd

import numpy as np

out_dict = pd.read_csv(os.path.join(os.getcwd(),'reports/training_RCNN_v2.csv'),index_col=0)
lowest_val_loss = out_dict['val_loss'].to_numpy()[-1]
out_dict = out_dict.to_dict()
for key in out_dict.keys():
    new_list = []
    for val in out_dict[key].values():
        new_list.append(val)
    
    out_dict[key] = new_list

# Make plot of training curves
fig, ax = plt.subplots(1,2,figsize=(15,7))
ax[0].plot(out_dict['train_acc'])
ax[0].plot(out_dict['val_acc'])
ax[0].set_xlabel('Epoch number')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Training and validation accuracy')
ax[0].axvline(x=7, color='black',linestyle='--')
ax[0].legend(('Train accuracy','Validation accuracy','Best model'))
ax[1].plot(out_dict['train_loss'])
ax[1].plot(out_dict['val_loss'])
ax[1].set_xlabel('Epoch number')
ax[1].set_ylabel('Loss')
ax[1].set_title('Training and validation loss')
ax[1].axvline(x=7, color='black',linestyle='--')
ax[1].legend(('Train loss','Validation loss','Best model'))

fig.savefig(os.getcwd()+'/reports/figures/training_curves/RCNN_v2.png')
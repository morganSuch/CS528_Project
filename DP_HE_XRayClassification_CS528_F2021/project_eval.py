import torch 
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np


x = np.arange(0, 138*21, 138)
print(len(x))
eps_high = [0.0, 0.10, 0.15, 0.18, 0.21, 0.24, 0.27, 0.29, 0.31, 
 	 		0.33, 0.35, 0.37, 0.39, 0.41, 0.42, 0.44, 0.45, 0.47, 
 	 		0.48, 0.50, 0.51]
eps_relax = [0.0, 3.73, 5.58, 7.09, 8.44, 9.67, 10.82, 11.91, 12.95, 13.95, 14.92,
			 15.87, 16.78, 17.68, 18.55, 19.41, 20.26, 21.09, 21.90, 22.71, 23.50]

plt.plot(x, eps_high, label='High Privacy')
plt.plot(x, eps_relax, label='Relaxed Privacy')
plt.legend()
plt.title('Privacy Budget Expended During Training')
plt.xlabel('Training Step')
plt.savefig('plots/privacy_budget.png')

baseline_train_loss = pd.read_csv("output/baseline_trainloss.csv")
baseline_val_loss = pd.read_csv("output/baseline_valloss.csv")
baseline_val_roc = pd.read_csv("output/baseline_valroc.csv")

sgd_train_loss = pd.read_csv("output/dp_sgd_trainloss.csv")
sgd_val_loss = pd.read_csv("output/dp_sgd_valloss.csv")
sgd_val_roc = pd.read_csv("output/dp_sgd_valroc.csv")

adam_train_loss = pd.read_csv("output/dp_adam_trainloss.csv")
adam_val_loss = pd.read_csv("output/dp_adam_valloss.csv")
adam_val_roc = pd.read_csv("output/dp_adam_valroc.csv")

adagrad_train_loss = pd.read_csv("output/dp_adagrad_trainloss.csv")
adagrad_val_loss = pd.read_csv("output/dp_adagrad_valloss.csv")
adagrad_val_roc = pd.read_csv("output/dp_adagrad_valroc.csv")

fig, axs = plt.subplots(2,2)
fig.set_size_inches(20, 12)

axs[0,0].plot(baseline_train_loss['Step'], baseline_train_loss['Value'], 
		color='blue', linestyle='solid',label='Training Loss')
axs[0,0].plot(baseline_val_loss['Step'], baseline_val_loss['Value'], 
		color='blue', linestyle='dashed', label='Validation Loss')
axs[0,0].plot(baseline_val_roc['Step'], baseline_val_roc['Value'], 
		color='blue', linestyle='dotted', label='Validation ROC')
axs[0,0].legend()
axs[0,0].set_ylim(0, 1.3)
axs[0,0].set_title('No Privacy Baseline')
axs[0,0].set_xlabel('Training Step')

axs[1,0].plot(sgd_train_loss['Step'], sgd_train_loss['Value'], 
		color='red', linestyle='solid',label='Training Loss')
axs[1,0].plot(sgd_val_loss['Step'], sgd_val_loss['Value'], 
		color='red', linestyle='dashed', label='Validation Loss')
axs[1,0].plot(sgd_val_roc['Step'], sgd_val_roc['Value'], 
		color='red', linestyle='dotted', label='Validation ROC')
axs[1,0].set_xlabel('Training Step')
axs[1,0].legend()
axs[1,0].set_ylim(0, 1.3)
axs[1,0].set_title('High Privacy SGD')

axs[0,1].plot(adam_train_loss['Step'], adam_train_loss['Value'], 
		color='green', linestyle='solid',label='Training Loss')
axs[0,1].plot(adam_val_loss['Step'], adam_val_loss['Value'], 
		color='green', linestyle='dashed', label='Validation Loss')
axs[0,1].plot(adam_val_roc['Step'], adam_val_roc['Value'], 
		color='green', linestyle='dotted', label='Validation ROC')
axs[0,1].set_ylim(0, 1.3)
axs[0,1].legend()
axs[0,1].set_title('High Privacy Adam')
axs[0,1].set_xlabel('Training Step')

axs[1,1].plot(adagrad_train_loss['Step'], adagrad_train_loss['Value'], 
		color='brown', linestyle='solid',label='Training Loss')
axs[1,1].plot(adagrad_val_loss['Step'], adagrad_val_loss['Value'], 
		color='brown', linestyle='dashed', label='Validation Loss')
axs[1,1].plot(adagrad_val_roc['Step'], adagrad_val_roc['Value'], 
		color='brown', linestyle='dotted', label='Validation ROC')
axs[1,1].set_ylim(0, 1.3)
axs[1,1].legend()
axs[1,1].set_title('High Privacy Adagrad')
axs[1,1].set_xlabel('Training Step')
fig.suptitle('High Privacy v. Baseline Training Curves (RDP Epsilon = 0.64)', fontsize=16)
plt.savefig('plots/train_highprivacy_loss.png')

baseline_train_loss = pd.read_csv("output/baseline_trainloss.csv")
baseline_val_loss = pd.read_csv("output/baseline_valloss.csv")
baseline_val_roc = pd.read_csv("output/baseline_valroc.csv")

sgd_train_loss = pd.read_csv("output/relaxmid_sgd_trainloss.csv")
sgd_val_loss = pd.read_csv("output/relaxmid_sgd_valloss.csv")
sgd_val_roc = pd.read_csv("output/relaxmid_sgd_valroc.csv")

adam_train_loss = pd.read_csv("output/relaxmid_adam_trainloss.csv")
adam_val_loss = pd.read_csv("output/relaxmid_adam_valloss.csv")
adam_val_roc = pd.read_csv("output/relaxmid_adam_valroc.csv")

adagrad_train_loss = pd.read_csv("output/relaxmid_adagrad_trainloss.csv")
adagrad_val_loss = pd.read_csv("output/relaxmid_adagrad_valloss.csv")
adagrad_val_roc = pd.read_csv("output/relaxmid_adagrad_valroc.csv")

fig, axs = plt.subplots(2,2)
fig.set_size_inches(20, 12)

axs[0,0].plot(baseline_train_loss['Step'], baseline_train_loss['Value'], 
		color='blue', linestyle='solid',label='Training Loss')
axs[0,0].plot(baseline_val_loss['Step'], baseline_val_loss['Value'], 
		color='blue', linestyle='dashed', label='Validation Loss')
axs[0,0].plot(baseline_val_roc['Step'], baseline_val_roc['Value'], 
		color='blue', linestyle='dotted', label='Validation ROC')
axs[0,0].legend()
axs[0,0].set_ylim(0, 1.3)
axs[0,0].set_title('No Privacy Baseline')
axs[0,0].set_xlabel('Training Step')

axs[1,0].plot(sgd_train_loss['Step'], sgd_train_loss['Value'], 
		color='red', linestyle='solid',label='Training Loss')
axs[1,0].plot(sgd_val_loss['Step'], sgd_val_loss['Value'], 
		color='red', linestyle='dashed', label='Validation Loss')
axs[1,0].plot(sgd_val_roc['Step'], sgd_val_roc['Value'], 
		color='red', linestyle='dotted', label='Validation ROC')
axs[1,0].set_xlabel('Training Step')
axs[1,0].legend()
axs[1,0].set_ylim(0, 1.3)
axs[1,0].set_title('Relaxed Privacy SGD')

axs[0,1].plot(adam_train_loss['Step'], adam_train_loss['Value'], 
		color='green', linestyle='solid',label='Training Loss')
axs[0,1].plot(adam_val_loss['Step'], adam_val_loss['Value'], 
		color='green', linestyle='dashed', label='Validation Loss')
axs[0,1].plot(adam_val_roc['Step'], adam_val_roc['Value'], 
		color='green', linestyle='dotted', label='Validation ROC')
axs[0,1].set_ylim(0, 1.3)
axs[0,1].legend()
axs[0,1].set_title('Relaxed Privacy Adam')
axs[0,1].set_xlabel('Training Step')

axs[1,1].plot(adagrad_train_loss['Step'], adagrad_train_loss['Value'], 
		color='brown', linestyle='solid',label='Training Loss')
axs[1,1].plot(adagrad_val_loss['Step'], adagrad_val_loss['Value'], 
		color='brown', linestyle='dashed', label='Validation Loss')
axs[1,1].plot(adagrad_val_roc['Step'], adagrad_val_roc['Value'], 
		color='brown', linestyle='dotted', label='Validation ROC')
axs[1,1].set_ylim(0, 1.3)
axs[1,1].legend()
axs[1,1].set_title('Relaxed Privacy Adagrad')
axs[1,1].set_xlabel('Training Step')
fig.suptitle('Relaxed Privacy v. Baseline Training Curves (RDP Epsilon = 19.16)', fontsize=16)
plt.savefig('plots/train_relaxed_loss.png')

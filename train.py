import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import random
from sklearn.model_selection import train_test_split

from datetime import datetime
import argparse

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="data/proc/preprocessed.npz", help="Path to source data", type=str)
parser.add_argument("--dest", default="models/model_"+datetime.now().strftime("%Y%m%d%H%M%S")+".pth", help="Path to save model after training", type=str)
parser.add_argument("--hidden", default=100, help="Size of the hidden layer", type=int)
parser.add_argument("--seed", default=102548, help="Random Seed", type=int)
parser.add_argument("--batch-size", default=256, help="Batch Size for Training", type=int)
parser.add_argument("--epochs", default=25, help="Number of Epochs", type=int)
parser.add_argument("--split", default=0.25, help="Train-test split", type=float)
parser.add_argument("--lr", default=0.001, help="Learning rate", type=float)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

data = np.load(args.src)
inputs = data['inputs'].astype(np.float32)
targets = data['targets'].astype(np.float32)
num_samples = len(inputs)
train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=args.split, random_state=args.seed)
input_size = train_inputs.shape[1]
target_size = train_targets.shape[1]
train_dataset = DataLoader(np.hstack([train_inputs, train_targets]), batch_size=args.batch_size, shuffle=True)
test_dataset = DataLoader(np.hstack([test_inputs, test_targets]), batch_size=args.batch_size, shuffle=False)

model = Model(input_size, target_size, args.hidden).to(device)
optimizer = torch.optim.AdamW(model.parameters(), args.lr)

def run(iterator):
	total_loss = 0
	for i, sample in enumerate(iterator):
		if model.training:
			optimizer.zero_grad()
		inputs = torch.Tensor(sample[:, :input_size]).to(device)
		targets = torch.Tensor(sample[:, input_size:]).to(device)
		preds = model(inputs)
		loss = F.mse_loss(preds, targets, reduction='sum')
		total_loss += loss#*len(inputs)
		if model.training:
			loss.backward()
			optimizer.step()

	return total_loss/len(iterator)

for e in range(args.epochs):
	model.train()
	train_loss = run(train_dataset).detach().cpu().numpy().item()
	model.eval()
	with torch.no_grad():
		test_loss = run(test_dataset).cpu().numpy().item()
	print('Epoch %0.2d: Training Loss %0.4e, Testing Loss %0.4e' %(e,train_loss,test_loss))

plot_inputs = inputs[:int(args.split*len(inputs))]
plot_targets = targets[:int(args.split*len(inputs))]
with torch.no_grad():
	preds = model(torch.Tensor(plot_inputs).to(device)).cpu().numpy()

observed_q = plot_inputs[:, :4]
desired_q = plot_inputs[:, 4:8]
desired_q_dot = plot_inputs[:, 8:12]
desired_q_ddot = plot_inputs[:, 12:16]
desired_tau = plot_inputs[:, 16:]

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

jet = cm = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=4)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# fig = plt.figure()
# ax_pos = fig.add_subplot(121)
# ax_pred = fig.add_subplot(122)
# ax_acc = fig.add_subplot(223)
# ax_tau = fig.add_subplot(224)


for i in range(4):
	# ax_pos.plot(observed_q[:, i], color=scalarMap.to_rgba(i), linestyle='--')
	# ax_pos.plot(desired_q[:, i], color=scalarMap.to_rgba(i), linestyle='-')

	plt.plot(plot_targets[:, i], color=scalarMap.to_rgba(i), linestyle='-')
	plt.plot(preds[:, i], color=scalarMap.to_rgba(i), linestyle='-', marker='.')
	plt.fill_between(np.arange(len(plot_targets)), plot_targets[:,i], preds[:, i], color=scalarMap.to_rgba(i), alpha = 0.7)
	
	# ax_vel.plot(desired_q_dot[:, i], color=scalarMap.to_rgba(i), linestyle='-')

	# ax_acc.plot(desired_q_ddot[:, i], color=scalarMap.to_rgba(i), linestyle='-')
	# ax_tau.plot(desired_tau[:, i], color=scalarMap.to_rgba(i), linestyle='-')
plt.show()
torch.save(model, args.dest)
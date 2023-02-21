import numpy as np
from numpy import genfromtxt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="data/raw/csv/", help="Path to source data dir", type=str)
parser.add_argument("--dest", default="data/proc/preprocessed.npz", help="Path to destination data dir", type=str)
parser.add_argument("--length", default=1.0, help="How much of each demonstration to keep (fraction of the length)", type=float)
parser.add_argument("--visualize", action="store_true", help="Whether to visualize the data or not")
args = parser.parse_args()

keys = ""
idxMap = {}
delimiter = ';'
dataset = []
observed_q = []
observed_q_dot = []
desired_q = []
desired_q_dot = []
desired_q_ddot = []
desired_tau = []

for path in os.listdir(args.src):
	filepath = os.path.join(args.src,path)
	if keys == "":
		with open(filepath) as f:
			keys = f.readline()[:-1].split(delimiter)
			idxMap = {keys[i]:i for i in range(len(keys))}
	my_data = genfromtxt(filepath, delimiter=delimiter)
	my_data = my_data[1:] # first row is nan from the labels
	
	last = int(args.length*len(my_data)) # sometimes the ending has high drift due to some issue on pepper during data collection, hence discarding the last of each run
	dataset.append(my_data[:last])
	
	observed_q.append(my_data[:, idxMap["qIn_6"]:idxMap["qIn_10"]][:last])
	observed_q_dot.append(my_data[:, idxMap["Observers_PepperObserver_Encoder_encoderFiniteDifferences_6"]:idxMap["Observers_PepperObserver_Encoder_encoderFiniteDifferences_10"]][:last])
	desired_q.append(my_data[:, idxMap["qOut_6"]:idxMap["qOut_10"]][:last])
	desired_q_dot.append(my_data[:, idxMap["qDotOut_6"]:idxMap["qDotOut_10"]][:last])
	desired_q_ddot.append(my_data[:, idxMap["qDDotOut_6"]:idxMap["qDDotOut_10"]][:last])
	desired_tau.append(my_data[:, idxMap["tauOut_6"]:idxMap["tauOut_10"]][:last])
	
dataset = np.vstack(dataset)
observed_q = np.vstack(observed_q)
observed_q_dot = np.vstack(observed_q_dot)
desired_q = np.vstack(desired_q)
desired_q_dot = np.vstack(desired_q_dot)
desired_q_ddot = np.vstack(desired_q_ddot)
desired_tau = np.vstack(desired_tau)

np.savez_compressed(args.dest, 
	inputs=np.hstack([desired_q_dot, desired_q_ddot, desired_tau]), 
	targets=desired_q - observed_q
)


if args.visualize:

	import matplotlib.pyplot as plt
	import matplotlib.colors as colors
	import matplotlib.cm as cmx

	jet = cm = plt.get_cmap('jet') 
	cNorm  = colors.Normalize(vmin=0, vmax=4)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

	fig = plt.figure()
	ax_pos = fig.add_subplot(221)
	ax_vel = fig.add_subplot(222)
	ax_acc = fig.add_subplot(223)
	ax_tau = fig.add_subplot(224)


	for i in range(4):
		ax_pos.plot(observed_q[:, i], color=scalarMap.to_rgba(i), linestyle='--')
		ax_pos.plot(desired_q[:, i], color=scalarMap.to_rgba(i), linestyle='-')
		
		ax_vel.plot(desired_q_dot[:, i], color=scalarMap.to_rgba(i), linestyle='-')
		ax_vel.plot(observed_q_dot[:, i], color=scalarMap.to_rgba(i), linestyle='--')

		ax_acc.plot(desired_q_ddot[:, i], color=scalarMap.to_rgba(i), linestyle='-')
		ax_tau.plot(desired_tau[:, i], color=scalarMap.to_rgba(i), linestyle='-')
	plt.show()

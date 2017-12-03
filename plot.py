#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

def plot(path):
	file = open("./logs/" + path, "r")
	lines = file.readlines()
	file.close()

	data = []
	for line in lines:
		line = line.strip()
		words = line.split()
		for w in range(len(words)):
			if words[w] == "15998/15998":
				data.append([float(words[w+6]), float(words[w+9]), \
					float(words[w+12]), float(words[w+15])])
	data = np.array(data)
	t = np.linspace(0, 100, 100)
	plt.subplot(2, 1, 1)
	train_err, = plt.plot(t, data[:,0], 'r', label="train_err")
	val_error, = plt.plot(t, data[:,2], 'g', label="val_err")
	first_legend = plt.legend(handles=[train_err], loc=4)
	ax = plt.gca().add_artist(first_legend)
	sec_legend = plt.legend(handles=[val_error], loc=1)
	ax2 = plt.gca().add_artist(sec_legend)
	# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

	plt.subplot(2, 1, 2)
	train_acc, = plt.plot(t, data[:,1], 'b', label="train_accuracy")
	val_acc, = plt.plot(t, data[:,3], 'y', label="val_accuracy")
	third_legend = plt.legend(handles=[train_acc], loc=1)
	ax3 = plt.gca().add_artist(third_legend)
	fourth_legend = plt.legend(handles=[val_acc], loc=4)
	ax4 = plt.gca().add_artist(fourth_legend)
	plt.savefig("./figures/"+path[:-4]+".png")
	plt.show()
	


if __name__ == '__main__':
	plot("gru_w2v_noconv.txt")
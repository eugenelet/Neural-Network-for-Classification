from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from matplotlib import cm
from neural import *

def plot_decision_boundary(weight, input, z, layer_num, neuron_num, activ_type):
	x = np.ravel(input[:,0])
	y = np.ravel(input[:,1])
	fig1 = plt.figure(1)
	plt.hold(True)

	x_s=np.arange(np.amin(x), np.amax(x), 0.01)                # generate a mesh
	y_s=np.arange(np.amin(y), np.amax(y), 0.01)


	x_surf, y_surf = np.meshgrid(x_s, y_s)

	xy=np.vstack((x_surf.flatten(),y_surf.flatten())).T

	logit = test_neural(weight, xy, layer_num, neuron_num, activ_type) #xy.dot(weight) # N x K
	boundary = np.zeros(xy.shape[0]) # N
	for i in range(xy.shape[0]):
		boundary[i] = np.argmax(logit[i]) 

	z_surf = np.zeros(np.shape(xy)[0])
	z_surf[np.where(boundary==0)] = 1
	z_surf[np.where(boundary==1)] = 2
	z_surf[np.where(boundary==2)] = 3
	z_surf = z_surf.reshape(x_surf.shape)

	plt.contourf(x_surf,y_surf,z_surf, 8, alpha=.75, cmap='jet')
	plt.scatter(x, y, s=20,c=z, marker = 'o', cmap = cm.jet );                        # plot a 3d scatter plot


	plt.show()
	return boundary

def plot_error(error, valid_error):
	plt.figure(2)
	error_plt, = plt.plot(error, label='Training Error')
	valid_error_plt, = plt.plot(valid_error, label='Validation Error')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.legend(handles=[error_plt,valid_error_plt])
	plt.title('Training Error and Validation Error vs Epoch')
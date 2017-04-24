from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from pca import *
from neural import *
from my_plot import *
from compute_accuracy import *
import sys


if size(sys.argv) < 3:
	print "python main.py [1l|2l] [sgd|mb]"
	sys.exit()
else:
	if sys.argv[1] == '1l': # True:Sigmoid False:ReLU
		activ_type = True 
	else:
		activ_type = False
	if sys.argv[2] == 'sgd':
		sgd_mb = True
	else:
		sgd_mb = False	
	if size(sys.argv) == 4:
		if sys.argv[3] == '1':
			update = True
		else:
			update = False
	else:
		update = False

pca_dim = 2

test_size = 100
imageFolderPath = 'Data_Train/Class1'
imagePath = glob.glob(imageFolderPath+'/*.bmp') 
train_size = len(imagePath) - test_size
im_array_1 = np.array( [np.array(Image.open(imagePath[i])) for i in range(train_size)] )
imageFolderPath = 'Data_Train/Class2'
imagePath = glob.glob(imageFolderPath+'/*.bmp') 
im_array_2 = np.array( [np.array(Image.open(imagePath[i])) for i in range(train_size)] )
imageFolderPath = 'Data_Train/Class3'
imagePath = glob.glob(imageFolderPath+'/*.bmp') 
im_array_3 = np.array( [np.array(Image.open(imagePath[i])) for i in range(train_size)] )


imageFolderPath = 'Data_Train/Class1'
imagePath = glob.glob(imageFolderPath+'/*.bmp') 
im_array_1_test = np.array( [np.array(Image.open(imagePath[i])) for i in range(len(imagePath) - test_size, len(imagePath))] )
imageFolderPath = 'Data_Train/Class2'
imagePath = glob.glob(imageFolderPath+'/*.bmp') 
im_array_2_test = np.array( [np.array(Image.open(imagePath[i])) for i in range(len(imagePath) - test_size, len(imagePath))] )
imageFolderPath = 'Data_Train/Class3'
imagePath = glob.glob(imageFolderPath+'/*.bmp') 
im_array_3_test = np.array( [np.array(Image.open(imagePath[i])) for i in range(len(imagePath) - test_size, len(imagePath))] )

total_array = np.vstack((np.vstack((im_array_1,im_array_2)),im_array_3))
total_array = total_array.reshape(train_size*3,900)
[red_array, pca_eigvec, pca_mean, pca_eigval] = pca(total_array,pca_dim)

target = np.zeros((train_size*3,3))
target[0:train_size,0] = 1
target[train_size:train_size*2,1] = 1
target[train_size*2:train_size*3,2] = 1

test_target = np.zeros((test_size*3,3))
test_target[0:test_size,0] = 1
test_target[test_size:test_size*2,1] = 1
test_target[test_size*2:test_size*3,2] = 1


# For plotting
plot_target = np.zeros(train_size*3)
plot_target[0:train_size] = 1
plot_target[train_size:train_size*2] = 2
plot_target[train_size*2:train_size*3] = 3

# Number of neurons on each layer
if activ_type == True:
	neuron_num = [red_array.shape[1] ,30, 3]
else:
	neuron_num = [red_array.shape[1] , 17, 12, 3]

layer_num = size(neuron_num)

# Project test data to basis of training data
total_array = np.vstack((np.vstack((im_array_1_test,im_array_2_test)),im_array_3_test))
total_array = total_array.reshape(test_size*3,900)
test_array = (total_array - pca_mean).dot(pca_eigvec) / pca_eigval**0.5

# For plotting
plot_target = np.zeros(test_size*3)
plot_target[0:test_size] = 1
plot_target[test_size:test_size*2] = 2
plot_target[test_size*2:test_size*3] = 3

batch_num = 10
# Pick between SGD or Mini-Batch
if sgd_mb == True:
	weight,error,valid_error= train_neural(red_array, target, layer_num, neuron_num, activ_type, \
		test_array, test_target, plot_target, update)
else:
	weight, error, valid_error = train_neural_mini_batch(red_array, target, layer_num, \
		neuron_num, activ_type, test_array, test_target, plot_target, batch_num, update)

# Plot Error Functions
error = np.ravel(np.array(error))
valid_error = np.ravel(np.array(valid_error))
plot_error(error, valid_error)

# Plot Decision Boundary
plot_decision_boundary(weight, test_array, plot_target, layer_num, neuron_num, activ_type)

# Test performance
test_res = test_neural(weight, test_array, layer_num, neuron_num, activ_type)
boundary = np.zeros(test_res.shape[0]) # N
for i in range(test_res.shape[0]):
    boundary[i] = np.argmax(test_res[i]) 
print compute_accuracy(boundary,np.shape(boundary)[0], 3)


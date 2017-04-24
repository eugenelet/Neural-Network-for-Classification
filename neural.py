import numpy as np
import math
import random
# from plot_decision_boundary import *
from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# To observe changes of boundary over time
plt.ion()

def train_neural(input, target, layer_num, neuron_num, activ_type, test_input, test_target, plot_target, update):
    # eta = 0.0005
    eta = 0.003
    input_layer = np.column_stack((input,np.ones(np.shape(input)[0]))) # Add bias term
    test_input_layer = np.column_stack((test_input,np.ones(np.shape(test_input)[0]))) # Add bias term

    # Shuffle order for stochastic
    shuffle_index = list(zip(input_layer, target))
    random.shuffle(shuffle_index)
    input_layer, target = zip(*shuffle_index)

    # Convert to array
    input_layer = np.array(input_layer)
    target = np.array(target)

    # Initial weight with random values
    weight = [] # create empty list
    for layer in range(layer_num - 1):
        weight.append(np.random.rand(neuron_num[layer] + 1, neuron_num[layer+1]) / math.sqrt(neuron_num[layer]))

    # Store error for plot
    accu_error = []
    accu_valid_error = []

    # Initial Error
    error = 0
    min_error = 999999

    # Start stochastic
    for sample_num in range(input.shape[0]*50): # For all sample data (Stochastic)
        current_weight = np.copy(weight)

        # Forward and Backward
        output, activ_layer = forward(input_layer[sample_num%input.shape[0]],weight,layer_num,neuron_num,activ_type,3)
        weight = backward(weight,target[sample_num%input.shape[0]],output,activ_layer,eta,layer_num,activ_type)

        # Update Error
        error = error - target[sample_num%input.shape[0]].dot(np.log(output).T)

        if sample_num!=0 and (sample_num%(input.shape[0]+1))==0:
            # Predict output of validation data
            output, activ_layer = forward(test_input_layer,current_weight,layer_num,neuron_num,activ_type,3)
            
            if update:
                plot_decision_boundary_non_block(weight, test_input, plot_target, layer_num, neuron_num, activ_type)

            # Compute validation error
            valid_error = - np.multiply(test_target, np.log(output))
            valid_error = np.sum(valid_error) / valid_error.shape[0]

            # Update only if new min valid error is found
            if valid_error < min_error:
                opt_weight = current_weight
                min_error = valid_error
            else: # Error bounces off minima, try to reduce learning rate
                weight = opt_weight
                eta = eta / 2
            error = error / target.shape[0] # Normalize

            # For plotting purpose
            accu_error.append(error)
            accu_valid_error.append(valid_error)
            print error
            print valid_error
            error = 0


    # return weight
    return opt_weight, accu_error, accu_valid_error


def train_neural_mini_batch(input, target, layer_num, neuron_num, activ_type, test_input, test_target, plot_target,\
 batch_num, update):
    # eta = 0.0005
    eta = 0.008
    input_layer = np.column_stack((input,np.ones(np.shape(input)[0]))) # Add bias term
    test_input_layer = np.column_stack((test_input,np.ones(np.shape(test_input)[0]))) # Add bias term

    # Shuffle order for stochastic
    shuffle_index = list(zip(input_layer, target))
    random.shuffle(shuffle_index)
    input_layer, target = zip(*shuffle_index)

    # Convert to array
    input_layer = np.array(input_layer)
    target = np.array(target)

    weight = [] # create empty list
    for layer in range(layer_num - 1):
        weight.append(np.random.rand(neuron_num[layer] + 1, neuron_num[layer+1]) / math.sqrt(neuron_num[layer]))

    # For plotting
    accu_error = []
    accu_valid_error = []

    error = 0 # Initial Value
    min_error = 999999
    for sample_num in range(0,input.shape[0]*500,input.shape[0]/batch_num): # For all sample data (Stochastic)
        current_weight = np.copy(weight)
        ind = sample_num%input.shape[0]
        grad_error_accu_total = []
        for batch_ind in range(input.shape[0]/batch_num):
            # forward and backward
            output, activ_layer = forward(input_layer[ind+batch_ind],weight,layer_num,neuron_num,activ_type,3)
            grad_error_accu = backward_mini_batch(weight,target[ind+batch_ind],output,activ_layer,eta,layer_num,activ_type)
            grad_error_accu_total.append(grad_error_accu)
        
        # Accumulate gradient of error from backpropagation
        grad_error = grad_error_accu_total[0] # first data
        for i in range(1,len(grad_error_accu_total)):
            for j in range(len(grad_error)):
                grad_error[j] = grad_error[j] + grad_error_accu_total[i][j]
        # Normalize Grad Error
        for j in range(len(grad_error)):
            grad_error[j] = grad_error[j] / len(grad_error_accu_total)
        # Update Weight
        for layer in range(len(grad_error)):
            weight[layer] = weight[layer] - eta*grad_error[layer]

        # Forward a single data to obtain training error
        output, activ_layer = forward(input_layer[ind],weight,layer_num,neuron_num,activ_type,3)
        error = error - target[ind].dot(np.log(output).T)

        if sample_num!=0 and (sample_num%(input.shape[0]))==0:
            # Forward validation data to compute validation error
            output, activ_layer = forward(test_input_layer,current_weight,layer_num,neuron_num,activ_type,3)
            if update:
                plot_decision_boundary_non_block(weight, test_input, plot_target, layer_num, neuron_num, activ_type)
            valid_error = - np.multiply(test_target, np.log(output))
            valid_error = np.sum(valid_error) / valid_error.shape[0]
            if valid_error < min_error:
                opt_weight = current_weight
                min_error = valid_error
            else:
                weight = opt_weight
                eta = eta / 2
            error = error / batch_num
            accu_error.append(error)
            accu_valid_error.append(valid_error)
            print error
            print valid_error
            error = 0
    return opt_weight, accu_error, accu_valid_error

def test_neural(weight, input, layer_num, neuron_num, activ_type):
    input_layer = np.column_stack((input,np.ones(np.shape(input)[0]))) # Add bias term
    output, activ_layer = forward(input_layer,weight,layer_num,neuron_num,activ_type,3)
    return output



def forward(input, weight, layer_num, neuron_num, activ_type, num_class):
    activ_layer = []
    current_layer = input # D
    activ_layer.append(current_layer)
    for layer in range(layer_num - 1):
        layer_out = current_layer.dot(weight[layer]) # 1 x M
        if layer<(layer_num-2): # Sigmoid (Leave last layer for Softmax)
            if activ_type==True:
                layer_out = sigmoid(layer_out)
            else:
                layer_out = ReLU(layer_out)
            # Add bias term
            layer_out = np.column_stack((layer_out,np.ones(layer_out.shape[0])))
            activ_layer.append(layer_out)
        # print layer_out.shape
        current_layer = layer_out
    # Softmax
    exp_max = np.amax(layer_out) # prevent overflow
    denom = sum(np.exp(layer_out.T - exp_max))
    for _class in range(num_class):
        current_layer.T[_class] = np.exp(current_layer.T[_class] - exp_max) / denom
    return current_layer, activ_layer
            

def backward(weight, target, output, activ_layer, eta, layer_num, activ_type):
    delta = output - target # K
    for layer in range(layer_num-2, -1, -1):
        grad_error = activ_layer[layer].T.dot(delta) # M x K
        weight[layer] = weight[layer] - eta*grad_error # M x K
        if activ_type == True:
            delta = np.multiply(np.multiply(activ_layer[layer], (1 - activ_layer[layer])) , weight[layer].dot(delta.T).T) # 1 x K
        else:
            grad_ReLU = activ_layer[layer]
            grad_ReLU[grad_ReLU>0] = 1
            grad_ReLU[grad_ReLU<=0] = 0
            delta = np.multiply(grad_ReLU, weight[layer].dot(delta.T).T) # 1 x K
        delta = delta[:,0:(delta.shape[1] - 1)]
    return weight


def backward_mini_batch(weight, target, output, activ_layer, eta, layer_num, activ_type):
    delta = output - target # K
    grad_error_accu = []
    for layer in range(layer_num-2, -1, -1):
        grad_error = activ_layer[layer].T.dot(delta) # M x K
        grad_error_store = np.copy(grad_error)
        grad_error_accu.insert(0,grad_error_store)
        if activ_type == True:
            delta = np.multiply(np.multiply(activ_layer[layer], (1 - activ_layer[layer])) , weight[layer].dot(delta.T).T) # 1 x K
        else:
            grad_ReLU = activ_layer[layer]
            grad_ReLU[grad_ReLU>0] = 1
            grad_ReLU[grad_ReLU<=0] = 0
            delta = np.multiply(grad_ReLU, weight[layer].dot(delta.T).T) # 1 x K
        delta = delta[:,0:(delta.shape[1] - 1)]
    return grad_error_accu

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

# To observe how boundary changes over time
def plot_decision_boundary_non_block(weight, input, z, layer_num, neuron_num, activ_type):
    x = np.ravel(input[:,0])
    y = np.ravel(input[:,1])
    fig1 = plt.figure(1)
    # plt.hold(True)

    # weight = np.column_stack((np.column_stack((weight[0:3],weight[3:6])),weight[6:9]))
    x_s=np.arange(np.amin(x), np.amax(x), 0.1)                # generate a mesh
    y_s=np.arange(np.amin(y), np.amax(y), 0.1)


    x_surf, y_surf = np.meshgrid(x_s, y_s)

    xy=np.vstack((x_surf.flatten(),y_surf.flatten())).T
    # xy = np.column_stack((np.ones(np.shape(xy)[0]), xy))

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

    plt.draw()
    plt.pause(0.001)
    return boundary

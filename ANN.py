import numpy as np
from scipy.special import expit   #for sigmoid function

class Layer:
    #attributes
    #weights, bias arrays
    weights = None  
    bias = None #for each nueron in the layer

    z = None    
    activations = None

    dW = 0 #dE/dW
    dB = 0 #dE/dB

    #constructor
    def __init__(self, neurons, activation_func) -> None:
        self.neurons = neurons #number of neurons in the layer
        self.g = activation_func #activation function for this layer
            #calling the activation function: self.g(x)
            #sigmoid, ReLU, or softmax
        
        #assign the activation function derivative
        if self.g == sigmoid :
            self.g_derivative = sigmoid_derivative
        elif self.g == ReLU :
            self.g_derivative = ReLU_derivative
        else: #softmax
            self.g_derivative = None


#activation functions and derivatives
def sigmoid(x) :
    return expit(x)

def sigmoid_derivative(x):  
    return expit(x)*(1-expit(x))

def ReLU(x):
    return np.maximum(0,x)

def ReLU_derivative(x):
    return x > 0
#returns false for negative numbers and 0

def softmax(x) :
    exp = np.exp(x - np.max(x))
    return exp / exp.sum(axis=0)



###########################################################################



class Network :
    #constructor
    def __init__(self, layers) -> None:
        self.layers = layers #list of layers
            # i=0 --> input layer
            # i=(n-1) --> output layer
        self.n = len(layers) #number of layers in the network

        #initialise weights and biases for all layers except input layer
        for i in range(1,self.n) :
            layer = self.layers[i]
            prev_layer = self.layers[i-1]

            layer.weights = np.random.rand(layer.neurons, prev_layer.neurons)  - 0.5
            layer.bias = np.random.rand(layer.neurons, 1)  - 0.5

    
    #training
    def train(self, X, Y, lr, epochs) : 
        expected_output = one_hot(Y)
        #input layer
        self.layers[0].activations = X #input activations


        #m = number of training observations
        r,m = X.shape

        #training episodes
        for _ in range(epochs) :
            #### forward propagation ##########################
            #for each layer
            for i in range(1,self.n) :
                layer = self.layers[i]
                prev_layer = self.layers[i-1]

                layer.z = np.dot(layer.weights, prev_layer.activations) + layer.bias
                layer.activations = layer.g(layer.z) # g = activation function


            #### back propagation #################################
            #loss function J = (predicted output - expected output)^2
            output_layer = self.layers[self.n-1]
            predicted_output = output_layer.activations
            error = 2*(predicted_output - expected_output) 

            #gradient for output layer
            #currently, prev_layer is the last hidden layer
            output_layer.dW = np.dot(error, prev_layer.activations.T)/m
            output_layer.dB = np.sum(error, 1)/m 

            #calculate gradient for each of the hidden layers
            for i in range(self.n-2, 0, -1) : #going from n-2 to 0 
                prev_layer = self.layers[i-1]
                layer = self.layers[i]
                next_layer = self.layers[i+1]

                #update the value of error for this layer
                error = np.dot(next_layer.weights.T, error)*layer.g_derivative(layer.z)
                    #error = dot(next layer weights, next layer error) * g'(z)

                #gradients
                layer.dW = np.dot(error , prev_layer.activations.T) / m
                layer.dB = np.sum(error, 1)/m
            
            #update the weights and biases
            for i in range(1, self.n):
                layer = self.layers[i]
                # formula: w = w - lr*dW
                layer.weights -= lr*(layer.dW)
                layer.bias -= (lr)*np.reshape(layer.dB, (layer.neurons, 1))
        #training completed
    

    #test
    def test(self, X, Y) :
        activations = X #input layer activation
        expected_output = Y

        ######forward propagation only ########
        #for each layer except input layer
        for i in range(1,self.n) :
            layer = self.layers[i]

            z = np.dot(layer.weights, activations) + layer.bias
            activations = layer.g(z)
            #no need to store the activations, z for each layer
            #z, activation of only the output layer matters
        #now, "activations" has the value of the output layer activations

        #### calculate accuracy #######
        predicted_output = np.argmax(activations, 0)

        #print predicted , expected output
        print("Expected output: ", expected_output)
        print("predicted output: ",predicted_output)

        accuracy = np.sum(predicted_output == expected_output)/ expected_output.size
        return accuracy



#one hot encoding
def one_hot(Y) :
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
 
    return one_hot_Y





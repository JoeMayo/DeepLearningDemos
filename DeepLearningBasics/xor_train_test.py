 
import numpy as np
import backprop

def load_data(csv_file):
    X_orig = np.genfromtxt(csv_file, usecols=(0,1), delimiter=',')
    Y_orig = np.genfromtxt(csv_file, usecols=2, delimiter=',')
    X = X_orig.reshape(X_orig.shape[0], -1).T
    Y = Y_orig.reshape(1, Y_orig.shape[0])
    print("X.shape = " + str(X.shape))
    print("Y.shape = " + str(Y.shape))
    return X, Y

def param_shape(parameters):
    print("W1 = " + str(parameters["W1"].shape))
    print("b1 = " + str(parameters["b1"].shape))
    print("W2 = " + str(parameters["W2"].shape))
    print("b2 = " + str(parameters["b2"].shape))

def train():
    X, Y = load_data('xor_train.csv')
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    num_iterations = 25000
    np.random.seed(3)
    
    parameters = backprop.initialize_parameters(n_x, n_h, n_y)

    param_shape(parameters)

    for i in range(0, num_iterations):
         
        A2, cache = backprop.forward_propagate(X, parameters)
        
        cost = backprop.compute_cost(A2, Y, parameters)
 
        grads = backprop.backward_propagate(parameters, cache, X, Y)
 
        parameters = backprop.update_parameters(parameters, grads, learning_rate=1.2)
        
        if i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def test(parameters):
    X, Y = load_data('xor_test.csv')
    A2, _ = backprop.forward_propagate(X, parameters)
    predictions = np.where(A2[0]>0.5, 1, 0)
    
    print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

def main():
    parameters = train()
    param_shape(parameters)
    test(parameters)

if __name__ == "__main__":
    main()
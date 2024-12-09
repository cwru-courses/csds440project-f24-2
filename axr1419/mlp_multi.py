import numpy as np
import pickle

class MLP:
    # Defining the parameters of MLP
    def __init__(self, input_size=30, n_classes=3):
        # Match the size of input layer with the input data dememsion
        self.input_size = input_size
        self.n_hdn1 = 1024
        self.n_hdn2 = 512
        # Match the output layer size with the No of Classess
        self.n_output_layer = n_classes
        # Intialize the weights and Biases
        self.w_1 = np.random.randn(self.input_size, self.n_hdn1) * 0.01
        self.b_1 = np.zeros(self.n_hdn1)
        self.w_2 = np.random.randn(self.n_hdn1, self.n_hdn2) * 0.01
        self.b_2 = np.zeros(self.n_hdn2)
        self.w_3 = np.random.randn(self.n_hdn2, self.n_output_layer) * 0.01
        self.b_3 = np.zeros(self.n_output_layer)
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        # Use this for stability for numerical data
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def softmax_derivative(self, output):
        # Derivative of Softmax isn't simple, better take care while loss calulation
        return output * (1 - output)

    def feed_forward_block(self, input_mtrx):
        # Feed forward block Input : Input Matrix , Output: Predicted Class Label
        self.x = input_mtrx

        # Input layer
        self.z1 = np.dot(self.x, self.w_1) + self.b_1
        self.a1 = self.relu(self.z1)

        # First Hidden layer
        self.z2 = np.dot(self.a1, self.w_2) + self.b_2
        self.a2 = self.relu(self.z2)

        # Second Hidden Layer
        self.z3 = np.dot(self.a2, self.w_3) + self.b_3
        # Softmax is used for noormalize the probabilities across output layer
        self.a3 = self.softmax(self.z3) 
        return self.a3

    def backpropagation(self, y_true, learning_rate=0.01):
        """
        Backpropagation for Gradient calculation and weight updation using cross-entropy loss..
        Args:
            y_true: Labels of True Class for samples in encode form of shape(n_samples, n_classes).
            learning_rate: Learning rate 
        """
        m = y_true.shape[0]

        # Calculate the gradient of the output layer
        grad_z3 = self.a3 - y_true  # Derivative of softmax cross-entropy loss
        grad_w3 = np.dot(self.a2.T, grad_z3) / m
        grad_b3 = np.sum(grad_z3, axis=0) / m

        # Gradients for the second hidden layer
        grad_a2 = np.dot(grad_z3, self.w_3.T)
        grad_z2 = grad_a2 * self.relu_derivative(self.z2)
        grad_w2 = np.dot(self.a1.T, grad_z2) / m
        grad_b2 = np.sum(grad_z2, axis=0) / m

        # Gradients for the first hidden layer
        grad_a1 = np.dot(grad_z2, self.w_2.T)
        grad_z1 = grad_a1 * self.relu_derivative(self.z1)
        grad_w1 = np.dot(self.x.T, grad_z1) / m
        grad_b1 = np.sum(grad_z1, axis=0) / m

        # Update weights and biases
        self.w_3 -= learning_rate * grad_w3
        self.b_3 -= learning_rate * grad_b3
        self.w_2 -= learning_rate * grad_w2
        self.b_2 -= learning_rate * grad_b2
        self.w_1 -= learning_rate * grad_w1
        self.b_1 -= learning_rate * grad_b1

    # Save the weights with given file_name.pkl
    def save_weights(self, filepath):
        weights = {
            "w_1": self.w_1, "b_1": self.b_1,
            "w_2": self.w_2, "b_2": self.b_2,
            "w_3": self.w_3, "b_3": self.b_3
        }
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Weights saved to {filepath}")
    
    # Load the already saved model weights
    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        self.w_1 = weights["w_1"]
        self.b_1 = weights["b_1"]
        self.w_2 = weights["w_2"]
        self.b_2 = weights["b_2"]
        self.w_3 = weights["w_3"]
        self.b_3 = weights["b_3"]
        print(f"Weights loaded from {filepath}")

   
    def extract_features(self, input_mtrx):
        """
        # Extract the feature of from second hidden layer output for classification using TSVM hidden layer.
        Args:
            input_mtrx: Input array of features of shape (n_samples, n_features).
        Returns:
            Array of extraxted features from Second Hidden Layer (a2).
        """

        # Feed Forward Pass upto Second Hidden Layer
        self.x = input_mtrx
        self.z1 = np.dot(self.x, self.w_1) + self.b_1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w_2) + self.b_2
        self.a2 = self.relu(self.z2)
        return self.a2

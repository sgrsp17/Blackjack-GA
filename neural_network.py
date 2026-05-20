import numpy as np

class NeuralNetwork:
    def __init__(self, weights):
        """
        Initializes the Neural Network with weights derived from the GA DNA.
        DNA Size expected: 81
        Architecture:
        - Input Layer: 3 neurons (Player Sum, Dealer Card, Usable Ace)
        - Hidden Layer: 16 neurons
        - Output Layer: 1 neuron (Hit or Stand)
        """
        self.weights = np.array(weights)
        
        # 1. Weights from Input (3) to Hidden (16) -> 3 * 16 = 48 weights
        self.W1 = self.weights[0:48].reshape((3, 16))
        
        # 2. Biases for Hidden Layer -> 16 weights
        self.b1 = self.weights[48:64]
        
        # 3. Weights from Hidden (16) to Output (1) -> 16 * 1 = 16 weights
        self.W2 = self.weights[64:80].reshape((16, 1))
        
        # 4. Bias for Output Layer -> 1 weight
        self.b2 = self.weights[80:]

    def relu(self, x):
        """Activation function for hidden layer (Rectified Linear Unit)"""
        return np.maximum(0, x)

    def sigmoid(self, x):
        """Activation function for output layer to squash values between 0 and 1"""
        # np.clip prevents overflow errors in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def predict(self, observation):
        """
        Forward pass: takes the observation from the environment and decides the action.
        """
        # observation is a tuple like (14, 10, False). Convert to numpy array of numbers
        # False becomes 0.0, True becomes 1.0
        x = np.array([float(observation[0]), float(observation[1]), float(observation[2])])
        
        # Step 1: Input to Hidden Layer (Dot product of Inputs * W1 + b1)
        z1 = np.dot(x, self.W1) + self.b1
        
        # Step 2: Apply ReLU Activation Function
        a1 = self.relu(z1)
        
        # Step 3: Hidden to Output Layer (Dot product of a1 * W2 + b2)
        z2 = np.dot(a1, self.W2) + self.b2
        
        # Step 4: Apply Sigmoid Activation Function
        a2 = self.sigmoid(z2)
        
        # Step 5: Convert output to Action (0 for Stand, 1 for Hit)
        # a2 is an array with a single value, e.g., [0.73]. Extract the value.
        output_val = a2[0]
        
        if output_val > 0.5:
            return 1 # Hit
        else:
            return 0 # Stand

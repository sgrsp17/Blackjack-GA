class NeuralNetwork:
    def __init__(self, weights):
        self.weights = weights
        # Your colleague will implement the matrix construction here

    def predict(self, observation):
        # Your colleague will implement the forward pass here.
        # For now, we return 0 (Stand) or 1 (Hit) randomly to test the pipeline.
        import random
        return random.choice([0, 1])

class NeuralNetwork:
    def __init__(self, weights):
        self.weights = weights
        # O seu colega irá implementar a construção das matrizes aqui

    def predict(self, observation):
        # O seu colega irá implementar o forward pass aqui.
        # Por agora, devolvemos 0 (Stand) ou 1 (Hit) aleatoriamente ou fixo para testar a pipeline.
        import random
        return random.choice([0, 1])

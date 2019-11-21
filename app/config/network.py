class NetworkConfig:

    def __init__(self, *, hidden_size=128, number_of_hidden_layers=2):
        self.hidden_size = hidden_size
        self.number_of_hidden_layers = number_of_hidden_layers

    def __str__(self):
        return f'Network(hidden_size={self.hidden_size}, number_of_hidden_layers={self.number_of_hidden_layers})'

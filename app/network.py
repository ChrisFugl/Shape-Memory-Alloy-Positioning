from torch import nn

class Network(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, number_of_hidden_layers, output_activation=None):
        """
        Creates a network.

        :param input_size: #neurons in input layer
        :param hidden_size: #neurons in hidden layers
        :param output_size: #neurons in output layer
        :param number_of_hidden_layers: #hidden layers
        """
        super(Network, self).__init__()
        layers = self.create_layers(input_size, hidden_size, output_size, number_of_hidden_layers)
        if output_activation is not None:
            layers.append(output_activation)
        self.sequential = nn.Sequential(*layers)

    def forward(self, input):
        return self.sequential(input)

    def create_layers(self, input_size, hidden_size, output_size, number_of_hidden_layers):
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(number_of_hidden_layers):
            hidden_layer = self.hidden_layer(hidden_size)
            for item in hidden_layer:
                layers.append(item)
        layers.append(nn.Linear(hidden_size, output_size))
        return layers

    def hidden_layer(self, hidden_size):
        return [
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        ]

from keras.models import Sequential
from keras.layers import Dense
from keras import activations

def build_single_layer_regression_model(n_hidden):
    """Builds a single layer regression neural network.

    Layers:
        - Dense, n_hidden neurons, tanh activation
        - Dense, 1 neuron, linear activation

    Arguments:
        n_hidden(int): The number of hidden units in
                       the hidden layer.

    Returns:
        A keras Sequential model with the single layer
        neural network.
    """
    model = Sequential()
    model.add(Dense(n_hidden, activation='tanh', input_dim=2))
    model.add(Dense(1, activation='linear'))
    return model

def build_deep_regression_model(n_hidden, n_layers):
    """Builds a deep regression neural network.

    Layers:
        - Dense, n_hidden neurons, tanh activation (x n_layers)
        - Dense, 1 neuron, linear activation

    Arguments:
        n_hidden(int): The number of hidden units in
                       each hidden layer.
        n_layers(int): The number of hidden layers (does
                       not include the output layer).

    Returns:
        A keras Sequential model with the deep neural network.
    """

    model = Sequential()
    model.add(Dense(n_hidden, activation='tanh', input_dim=2))
    for _ in range(n_layers-1):
        model.add(Dense(n_hidden, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    return model
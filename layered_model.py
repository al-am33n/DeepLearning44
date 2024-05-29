import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.datasets import mnist

def define_dense_model_single_layer(input_length, activation_f='sigmoid', output_length=1):
    """
    Define a dense model with a single layer.

    Parameters:
    input_length (int): The number of inputs
    activation_f (str): The activation function for the layer
    output_length (int): The number of outputs (number of neurons)

    Returns:
    model (Sequential): The defined Keras model
    """
    # Create a Sequential model
    model = Sequential([
        # Define a dense layer with the specified activation function and input shape
        Input(shape=(input_length,)),
        Dense(output_length, activation=activation_f)
    ])
    return model

def define_dense_model_with_hidden_layer(input_length, 
                                         activation_func_array=['relu', 'sigmoid'],
                                         hidden_layer_size=10,
                                         output_length=1):
    """
    Define a dense model with a hidden layer.

    Parameters:
    input_length (int): The number of inputs
    activation_func_array (list): Activation functions for the hidden layer and the output layer
    hidden_layer_size (int): The number of neurons in the hidden layer
    output_length (int): The number of outputs (number of neurons in the output layer)

    Returns:
    model (Sequential): The defined Keras model
    """
    # Create a Sequential model
    model = Sequential([
        # Define the hidden layer with the specified activation function and input shape
        Input(shape=(input_length,)),
        Dense(hidden_layer_size, activation=activation_func_array[0]),
        # Define the output layer with the specified activation function
        Dense(output_length, activation=activation_func_array[1])
    ])
    return model

def get_mnist_data():
    """
    Get the MNIST data.

    Returns:
    (tuple): Training and testing data
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize and reshape the training data
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    # Normalize and reshape the testing data
    x_test = x_test.reshape(10000, 784).astype('float32') / 255 
    return (x_train, y_train), (x_test, y_test)

def binarize_labels(labels, target_digit=2):
    """
    Binarize the labels.

    Parameters:
    labels (np.array): Original labels
    target_digit (int): The digit to classify (default is 2)

    Returns:
    np.array: Binarized labels
    """
    # Binarize the labels: set to 1 if equal to target_digit, else 0
    labels = 1 * (labels == target_digit)
    return labels

def fit_mnist_model_single_digit(x_train, y_train, target_digit, model, epochs=10, batch_size=128):
    """
    Fit the model to the data.

    Parameters:
    x_train (np.array): Training data
    y_train (np.array): Training labels
    target_digit (int): The digit to classify
    model (Sequential): The Keras model to train
    epochs (int): Number of epochs to train (default is 10)
    batch_size (int): Batch size for training (default is 128)

    Returns:
    Sequential: The trained model
    """
    # Binarize the training labels based on the target digit
    y_train = binarize_labels(y_train, target_digit)
    # Compile the model with Adam optimizer and binary cross-entropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model with the specified number of epochs and batch size
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def evaluate_mnist_model_single_digit(x_test, y_test, target_digit, model):
    """
    Evaluate the model on the test data.

    Parameters:
    x_test (np.array): Test data
    y_test (np.array): Test labels
    target_digit (int): The digit to classify
    model (Sequential): The Keras model to evaluate

    Returns:
    (float, float): Loss and accuracy on the test data
    """
    # Binarize the test labels based on the target digit
    y_test = binarize_labels(y_test, target_digit)
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

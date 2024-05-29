from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from layered_model import define_dense_model_single_layer, define_dense_model_with_hidden_layer
from layered_model import fit_mnist_model_single_digit, evaluate_mnist_model_single_digit
from layered_model import binarize_labels, get_mnist_data

def test_define_dense_model_single_layer():
    model = define_dense_model_single_layer(43, activation_f='sigmoid', output_length=1)
    assert len(model.layers) == 1, "model should have 1 layer"
    assert model.input_shape == (None, 43), "input_shape is not correct"
    assert model.layers[0].output.shape == (None, 1), "output_shape is not correct"

def test_define_dense_model_with_hidden_layer():
    model = define_dense_model_with_hidden_layer(43, activation_func_array=['relu','sigmoid'], hidden_layer_size=11, output_length=1)
    assert len(model.layers) == 2, "model should have 2 layers"
    assert model.input_shape == (None, 43), "input_shape is not correct"
    assert model.layers[0].output.shape == (None, 11), "output_shape is not correct"
    assert model.layers[1].output.shape == (None, 1), "output_shape is not correct"

def test_fit_and_predict_mnist_single_digit_one_neuron():
    model = define_dense_model_single_layer(28*28, activation_f='sigmoid', output_length=1)
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    model = fit_mnist_model_single_digit(x_train, y_train, 2, model, epochs=5, batch_size=128)
    loss, accuracy = evaluate_mnist_model_single_digit(x_test, y_test, 2, model)
    assert accuracy > 0.9, "accuracy should be greater than 0.9"
    loss, accuracy = evaluate_mnist_model_single_digit(x_test, y_test, 3, model)
    assert accuracy < 0.9, "accuracy should be smaller than 0.9"

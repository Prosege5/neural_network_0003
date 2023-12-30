use rand::Rng;
use std::error::Error;
use std::fs::File;
use csv::{Reader, Writer};
use serde::Deserialize;

//An enum for neurons to set their activation function
enum ActivationFunction {
    Sigmoid,
    ReLU,
    Softmax,
}

//struct for neurons
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

//Impl for Neurons
impl Neuron {

}

//struct  for defining layers
struct Layer {
    neurons: Vec<Neuron>,
    activation_function: ,//set layer assigned function ***
}

impl Layer{

}

//struct for defining neural network
struct NeuralNetwork {
    layers: Vec<Vec<layer>>,
    layers_weights: Vec<Vec<f64>>,
    layers_biases: Vec<Vec<f64>>,

}

impl NeuralNetwork {
    //a function for initializing a new network
    fn new_network() -> NeuralNetwork {

    }

    //function to initialize first random weights and biases
    fn rand_w_and_b() -> Vec<Vec<f64>, f64> {

    }

    //function to load weights and biases from training or csv
    fn load_w_and_b() -> Vec<Vec<f64>, f64> {

    }

    //function to train neural network
    fn train_network(epochs: usize, ) -> Vec<Vec<f64>, f64> {

    }

    //function to run an inference the network
    fn inference_network() -> Vec<f64> {

    }
}

//---main function---
fn main() {

    
}

//---functions---
fn sigmoid() {

}

fn relu() {

}

fn softmax() {

}

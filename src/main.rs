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
    activation: ActivationFunction,//set layer assigned function ***
}

impl Layer{
    //forward propagation of a layer
    fn forward_propagation(inputs: Vec<f64>) -> Vec<f64> {

    }
}

//struct for defining neural network
struct NeuralNetwork {
    layers: Vec<Layer>,

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
    //Training Data File
    let training_data: &str = "training_data.csv";
    //Model file name for loading or saving
    let model: &str = "fruit_classification_0001.json";

    //Create a new neural network, set size and layer functions
    let network: NeuralNetwork = NeuralNetwork::new_network();
    
    //train network

    //output training data

    //inference network

    //output prediction
}

//---functions---
fn sigmoid() {

}

fn relu() {

}

fn softmax() {

}

// Goal & Learning: Create a program to make layers of neurons
// Get input vec, clac feed forward with activation, return output vec

//* Crates */
use rand::Rng;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Write;
use csv::{Reader, Writer};
use serde::{ Deserialize, Serialize };
use serde_json;

//* Enums */
// Activation function enum for Layer struct
enum Activation_Function {
    Sigmoid,
    ReLU,
    Softmax,
    None,
}
// Layer type enum for Layer struct
enum Layer_Type {
    Input,
    Hidden,
    Output,
}

//* Structs and Methods */
//Neuron struct and methods
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}
impl Neuron {
    //fn to create a new neuron
    fn new_neuron() -> Neuron {

    }
    //generate random weights and bias
    fn gen_w_and_b(&mut self) {

    }
    //update weights and bias?? used with backward propagation??
    fn update_w_and_b(&mut self) {

    }
}
//Layer struct and methods
struct Layer {
    type: Layer_Type,
    layer_size: usize,
    activation: Activation_Function,
    neurons: Vec<Neuron>,
}
impl Layer {
    //fn to create a new layer
    fn new_layer(type: Layer_Type, layer_size: usize, activation: Activation_Function) -> Layer {

    }
    // forward propagation
    fn forward_propagation(&self, inputs: Vec<f64>) -> Vec<f64> {

    }
    // Sigmoid Function
    fn sigmoid(output: f64) -> f64 {

    }
    // ReLU Function
    fn relu(output: f64) -> f64 {

    }
    // Softmax Function
    fn softmax(output: f64) -> f64 {

    }
    ////save model
    fn save_layer() -> {

    }
    ////load model
    fn load_layer() -> {

    }
}

//* Main Function */
fn main() {
    //create inputs vec from csv file
    let inputs: Vec<Vec<f64>> = load_inputs("inputs.csv", 3);

    //create a inputs layer
    let mut input_layer: Layer = Layer::new_layer(Layer_Type::Input, 3, Activation_Function::None);
    //create a hidden layer
    let mut hidden_layer: Layer = Layer::new_layer(Layer_Type::Hidden, 3, Activation_Function::Sigmoid);
    //create a output layer
    let mut output_layer: Layer = Layer::new_layer(Layer_Type::Output, 3, Activation_Function::Softmax);

    //manually feed forward each layer to generate output
    let output: Vec<f64> = output_layer.forward_propagation(
        hidden_layer.forward_propagation(
            input_layer.forward_propagation()
        )
    );


}

//* Functions */
fn load_inputs(file_name: &str) -> Vec<Vec<f64>> {

}

//* Training methods - Back propagation */
    //Mean Squared Error (MSE) Loss
    fn mse() -> {

    }
    //Partial Derivative
    fn partial_derivative() -> {

    }
    //Stochastic Gradient Descent (SGD) 
    fn sgd() -> {
        
    }
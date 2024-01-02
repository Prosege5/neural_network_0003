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

}

//* Main Function */
fn main() {
    //create inputs vec from csv file
    let inputs: Vec<Vec<f64>> = load_inputs("inputs.csv");

    //create an inputs layer
    let mut input_layer: Layer = Layer::new_layer(Layer_Type::Input);
    //create a hidden layer
    let mut hidden_layer: Layer = Layer::new_layer(Layer_Type::Hidden);
    //create a output layer
    let mut output_layer: Layer = Layer::new_layer(Layer_Type::Output);
}

//* Functions */
fn load_inputs(file_name: &str) -> Vec<Vec<f64>> {

}
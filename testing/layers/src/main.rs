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

}
//Layer struct and methods
struct Layer {
    type: 
}
impl Layer {

}

//* Main Function */
fn main() {
    //create inputs vec from csv file
    let inputs: Vec<Vec<f64>> = load_inputs("inputs.csv");
}

//* Functions */
fn load_inputs(file_name: &str) -> Vec<Vec<f64>> {

}
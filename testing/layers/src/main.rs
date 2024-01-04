// Goal & Learning: Create a program to make layers of neurons
// Get input vec, clac feed forward with activation, return output vec

//* Crates */
use rand::Rng;
//use std::error::Error;
//use std::fs;
//use std::fs::File;
//use std::io::Write;
//use csv::{Reader, Writer};
//use serde::{ Deserialize, Serialize };
//use serde_json;

//* Enums */
#[derive(Debug)]
// Activation function enum for Layer struct
enum ActivationFunction {
    Sigmoid,
//    ReLU,
    Softmax,
    None,
}

#[derive(Debug)]
// Layer type enum for Layer struct
enum LayerType {
    Input,
    Hidden,
    Output,
}

#[derive(Debug)]
//* Structs and Methods */
//Neuron struct and methods
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}
impl Neuron {
    //fn to create a new neuron
    fn new_neuron(inputs: &Vec<f64>) -> Neuron {
        Neuron {
            weights: vec![1.0; inputs.len()],
            bias: 0.0,
        }
    }
    //generate random weights and bias
    fn gen_w_and_b(&mut self) {
        let mut rng = rand::thread_rng();
        let dec_places: u32 = 6;
        for i in 0..self.weights.len() {
            self.weights[i] = round_f64(rng.gen_range(-1.0..1.0), dec_places);
        }
        self.bias = round_f64(rng.gen_range(-5.0..5.0), dec_places);
    }
    //update weights and bias?? used with backward propagation??
//    fn update_w_and_b(&mut self) {
//
//    }
}

#[warn(dead_code)]
#[derive(Debug)]
//Layer struct and methods
struct Layer {
    layer_type: LayerType,
    input_size: usize,
    layer_size: usize,
    activation: ActivationFunction,
    neurons: Vec<Neuron>,
}
impl Layer {
    //fn to create a new layer
    fn new_layer(layer_type: LayerType, input_size: usize, layer_size: usize, activation: ActivationFunction) -> Layer {
        let neurons = match layer_type {
            LayerType::Input => (0..layer_size).map(|_| Neuron::new_neuron(&vec![1.0])).collect(),
            _ => (0..layer_size).map(|_| {
                let mut neuron = Neuron::new_neuron(&vec![0.0; input_size]);
                neuron.gen_w_and_b();
                neuron
            }).collect(),
        };
        
        Layer {
            layer_type,
            input_size,
            layer_size,
            activation,
            neurons,
        }
    }
    // forward propagation
//    fn forward_propagation(&self, inputs: Vec<f64>) -> Vec<f64> {
//
//    }
//    // Sigmoid Function
//    fn sigmoid(output: f64) -> f64 {
//
//    }
//    // ReLU Function
//    fn relu(output: f64) -> f64 {
//
//    }
//    // Softmax Function
//    fn softmax(output: f64) -> f64 {
//
//    }
//    ////save model
//    fn save_layer() -> {
//
//    }
//    ////load model
//    fn load_layer() -> {
//
//    }
}

//* Main Function */
fn main() {
    //create inputs vec from csv file
//    let inputs: Vec<Vec<f64>> = load_inputs("inputs.csv", 3);

    //create a inputs layer
    let input_layer: Layer = Layer::new_layer(LayerType::Input, 3, 3, ActivationFunction::None);
    //create a hidden layer
    let hidden_layer: Layer = Layer::new_layer(LayerType::Hidden, 3, 3, ActivationFunction::Sigmoid);
    //create a output layer
    let output_layer: Layer = Layer::new_layer(LayerType::Output, 3, 3, ActivationFunction::Softmax);

    //manually feed forward each layer to generate output
//    let output: Vec<f64> = output_layer.forward_propagation(
//        hidden_layer.forward_propagation(
//            input_layer.forward_propagation()
//        )
//    );
    println!("Input Layer: {:?}\n", input_layer);
    println!("Hidden Layer: {:?}\n", hidden_layer);
    println!("Output Layer: {:?}\n", output_layer);

}

//* Funct//ions */
//fn load_inputs(file_name: &str) -> Vec<Vec<f64>> {
//
//}

fn round_f64(value: f64, decimal_places: u32) -> f64 {
    let factor = 10f64.powi(decimal_places as i32);
    (value * factor).round() / factor
}

//* Training methods - Back propagation */
//Mean Squared Error (MSE) Loss
//fn mse() -> {
//
//}
//Partial Derivative
//fn partial_derivative() -> {
//
//}
//Stochastic Gradient Descent (SGD) 
//fn sgd() -> {
//        
//}
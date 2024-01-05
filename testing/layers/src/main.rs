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
#[derive(Debug, PartialEq)]
// Activation function enum for Layer struct
enum ActivationFunction {
    Sigmoid,
//    ReLU,
//    Softmax,
    None,
}

#[derive(Debug, PartialEq)]
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
    fn forward_propagation(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let outputs: Vec<f64> = self.neurons.iter().map(|neuron| {
            let neuron_output = neuron.weights.iter().zip(inputs.iter())
                .map(|(weight, input)| weight * input)
                .sum::<f64>() + neuron.bias;
            Layer::apply_activation(neuron_output, &self.activation)
        }).collect();

        if self.layer_type == LayerType::Output {
            Layer::softmax(&outputs).iter().map(|&x| round_f64(x, 6)).collect()
        } else {
            outputs
        }
    }

    //Apply Activation function
    fn apply_activation(output: f64, function: &ActivationFunction) -> f64 {
        match function {
            ActivationFunction::Sigmoid => Layer::sigmoid(output),
            ActivationFunction::None => output, 
        }
    }
    // Sigmoid Function
    fn sigmoid(output: f64) -> f64 {
        1.0 / (1.0 + (output).exp())
    }
//    // ReLU Function
//    fn relu(output: f64) -> f64 {
//
//    }
    // Softmax Function
    fn softmax(outputs: &Vec<f64>) -> Vec<f64> {
        let exp_sum: f64 = outputs.iter().map(|x| x.exp()).sum();
        outputs.iter().map(|x| x.exp() / exp_sum).collect()
    }
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
//    let inputs_csv: Vec<Vec<f64>> = load_inputs("inputs.csv", 3);
    let inputs_testing: Vec<f64> = vec![1.0, 0.9, 1.5];

    //create a inputs layer
    let input_layer: Layer = Layer::new_layer(LayerType::Input, 3, 3, ActivationFunction::None);
    //create a hidden layer
    let hidden_layer: Layer = Layer::new_layer(LayerType::Hidden, 3, 3, ActivationFunction::Sigmoid);
    //create a output layer
    let output_layer: Layer = Layer::new_layer(LayerType::Output, 3, 3, ActivationFunction::None);

    //manually feed forward each layer to generate output
    let output: Vec<f64> = output_layer.forward_propagation(
        &hidden_layer.forward_propagation(
            &input_layer.forward_propagation(&inputs_testing)
        )
    );

    //print layer structs
    println!("Input Layer: {:?}\n", input_layer);
    println!("Hidden Layer: {:?}\n", hidden_layer);
    println!("Output Layer: {:?}\n", output_layer);
    //print inputs
    println!("Inputs: {:?}\n", inputs_testing);
    //print outputs from feed forward
    println!("Output: {:?}\n", output);
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
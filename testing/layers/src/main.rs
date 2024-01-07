// Goal & Learning: Create a program to make layers of neurons
// Get input vec, clac feed forward with activation, return output vec

//* Crates */
use rand::Rng;
use std::error::Error;
//use std::fs;
//use std::fs::File;
//use std::io::Write;
use csv::ReaderBuilder;
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

//#[warn(dead_code)]
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
            Layer::softmax(&outputs).iter().map(|&x| round_f64(x, 2)).collect()
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
    //create inputs vec from csv file, first 3 rows, skip 0
    let inputs_csv = match load_inputs("inputs.csv", 3, 1) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("Failed to read CSV: {}", e);
            return;
        }
    };
    //shift input data by mean value
    let shifted_data: Vec<Vec<f64>> = shift_data(&inputs_csv);
    //create a vec of the probabilities from the csv file for training
    let probs_csv = match load_inputs("inputs.csv", 3, 4) {
        Ok(probs) => probs,
        Err(e) => {
            eprintln!("Failed to read CSV: {}", e);
            return;
        }
    };

    //outputs vector for training
    let mut outputs: Vec<Vec<f64>> = Vec::new();

    //create a inputs layer
    let input_layer: Layer = Layer::new_layer(LayerType::Input, 3, 3, ActivationFunction::None);
    //create a hidden layer
    let hidden_layer: Layer = Layer::new_layer(LayerType::Hidden, 3, 3, ActivationFunction::Sigmoid);
    //create a output layer
    let output_layer: Layer = Layer::new_layer(LayerType::Output, 3, 3, ActivationFunction::None);

    //print inputs and probabilities
    println!("\nLoaded Inputs: {:?}", inputs_csv);
    println!("Shifted Data: {:?}", shifted_data);
    println!("Probabilities: {:?}\n", probs_csv);
    //print layer structs
    println!("Input Layer: {:?}\n", input_layer);
    println!("Hidden Layer: {:?}\n", hidden_layer);
    println!("Output Layer: {:?}\n", output_layer);

    //manually feed forward through data table inputs_csv
    for input in shifted_data {
        let output = output_layer.forward_propagation(
            &hidden_layer.forward_propagation(
                &input_layer.forward_propagation(&input)
            )
        );
        outputs.push(output);
    }

    //print outputs from feed forward
    println!("Output: {:?}\n", outputs);

    //training the network functionality
    let mse_loss: Vec<f64> = mean_squared_error(&probs_csv, &outputs);
    println!("Loss: {:?}", &mse_loss);
}

//* Functions */
fn load_inputs(file_path: &str, num_columns: usize, skip_columns: usize) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;

    let mut data = Vec::new();

    for result in reader.records() {
        let record = result?;
        let mut row: Vec<f64> = Vec::new();

        for (i, field) in record.iter().enumerate() {
            if i < skip_columns {
                continue;
            }
            if i >= skip_columns + num_columns {
                break;
            }

            match field.parse::<f64>() {
                Ok(num) => row.push(num),
                Err(_) => return Err(format!("Invalid floa literal at column {}: '{}'", i + 1, field).into()),
            }
        }

        data.push(row);
    }

    Ok(data)
}

fn round_f64(value: f64, decimal_places: u32) -> f64 {
    let factor = 10f64.powi(decimal_places as i32);
    (value * factor).round() / factor
}

fn get_means_cols(table: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut means = vec![0.0; table.len()];
    let rows = table.len();

    for row in table {
        for (i, &value) in row.iter().enumerate() {
            means[i] += value;
        }
    }

    for mean in &mut means {
        *mean /= rows as f64;
    }

    means
}

fn shift_data(inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let means: Vec<f64> = get_means_cols(&inputs);

    inputs.into_iter()
        .map(|row| row.into_iter().enumerate()
            .map(|(i, value)| round_f64(value - means[i], 2))
            .collect())
        .collect()
}

//* Training methods - Back propagation */
//Mean Squared Error (MSE) Loss
fn mean_squared_error(pred_true: &Vec<Vec<f64>>, pred_guess: &Vec<Vec<f64>>) ->  Vec<f64> {
    let rows = pred_true.len();
    let cols = pred_true[0].len();
    let mut squared_diff_sum = vec![0.0; cols];

    for (sub_true, sub_guess) in pred_true.iter().zip(pred_guess.iter()) {
        for (i, (&val_true, &val_guess)) in sub_true.iter().zip(sub_guess.iter()).enumerate() {
            squared_diff_sum[i] += (val_true - val_guess).powi(2);
        }
    }

    squared_diff_sum.iter().map(|&sum| round_f64(sum / rows as f64, 2)).collect()
}

//Partial Derivative
//fn partial_derivative() -> {
//
//}
//Stochastic Gradient Descent (SGD) 
//fn sgd() -> {
//        
//}
//testing for creation of nodes in the neural network 0003 program
//Goal: make a node with random weights and biases

//crates
use rand::Rng;
//use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Write;
//use csv::{Reader, Writer};
use serde::{ Deserialize, Serialize };
use serde_json;

//* future enum for activation function - applied in layer struct */

#[derive(Debug)]
#[derive(Serialize)]
#[derive(Deserialize)]
// Node Struct
struct Node {
    weights: Vec<f64>,
    bias: f64,
}

// Impl Node methods
impl Node {
    //fn to create a new node
    fn new_node(inputs: &Vec<f64>) -> Node {
        Node {
            weights: vec![0.1; inputs.len()],
            bias: 1.0,
        }
    }
    //generate random weights and biases
    fn gen_w_and_b(&mut self) {
        let mut rng = rand::thread_rng();
        let dec_places: u32 = 6;
        for i in 0..self.weights.len() {
            self.weights[i] = round_f64(rng.gen_range(-1.0..1.0), dec_places);
        }
        self.bias = round_f64(rng.gen_range(-5.0..5.0), dec_places);
    }
    //feed forward
    fn feedforward(&self, inputs: &Vec<f64>) -> f64 {
        let mut output = 0.0;
        for i in 0..inputs.len() {
            output = output + (self.weights[i] * inputs[i]);
        }
        return output + self.bias;
    }
    ////save model
    fn save_model(&self, file_path: &str) -> Result<(), serde_json::Error> {
        let serialized = serde_json::to_string(&self)?;
        let mut file = File::create(file_path).expect("Unable to create file!");
        file.write_all(serialized.as_bytes()).expect("Unable to write data");
        Ok(())
    }
    ////load model
    fn load_model(file_path: &str) -> Result<Node, serde_json::Error> {
        let data = fs::read_to_string(file_path).map_err(serde_json::Error::io)?;
        serde_json::from_str(&data)
    }

}

fn main() {
    //create inputs
    let inputs: Vec<f64> = vec![1.0, 2.0, 3.0];

    //create a new node
    let mut node1: Node = Node::new_node(&inputs);
    //generate nodes random weights and biases
    node1.gen_w_and_b();

    //feed forward node
    let node1_out: f64 = round_f64(node1.feedforward(&inputs), 6); 

    //save the model
    if let Err(e) = node1.save_model("node1.json") {
        eprintln!("Failed to write to JSON: {}", e);
    }

    //load a model
    let node2 = match Node::load_model("node1.json") {
        Ok (node) => {
            node
        },
        Err(e) => {
            eprintln!("Failed to load from JSON: {}", e);
            return;
        }
    };

    println!("\nInputs: {:?}", inputs);
    println!("Node-1: {:?}", node1);
    println!("Node-1 Out: {}\n", node1_out);
    println!("Node-2: {:?}", node2);

}


//functions
fn round_f64(value: f64, decimal_places: u32) -> f64 {
    let factor = 10f64.powi(decimal_places as i32);
    (value * factor).round() / factor
}

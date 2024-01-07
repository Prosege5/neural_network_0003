# neural_network_0003
Learning Rust and Neural Networks (WIP)

# Project info for 0003
- goal: train a network for a simple fruit classification dataset
- setup a framework for creating and training networks
- layed out code to be filled in (WIP)

# Learning Updates:
- Day 4: Stepping back and learning more
[file: /testing/nodes]
Today I stepped back to learn more about how to structure and plan a program in Rust.
I can make neurons with dynamic weight vectors based on input vec size.
Generating initial random weights and biases function
Feed forward functionality, no activation yet.
Saving a "model" to a .json file.
Loading a "model" from a .json to create a new neuron.

- Day 5: Program structuring setup for learning about layer creation better.
Created /testing/layers to create a program for dynamically making layers filled with neurons.
Today was mostly just setting up the framework and structure. 
I filled in quite a bit to get some idea of the flow and functionality I want.
Also, might start messing around with backward propagation with this program once I can fully feed forward an output.

- Day 6: More program planning an laying out in /testing/layers program.
Decided to make inputs a layer with an activation function of None. I will set 1 weight to 1 and bias to 0 to feed forward original input. Doing this method since I want to be able to save the layers in .json format and load from .json file.
Backpropagation functions in site for learning more on how it works once I can start getting outputs from the layers.

- Day 7: Progress has been made on inplementing functionality into neurons and layers in /testing/layers program.
Today I wrote the code to create new neurons, set their random weights and bias.
In the new layer method I check the layer type to setup an input layer out of neurons. I chose this method for when I setup layer loading and saving to and from .json format.
Lots of code lines commented out to start testing initial outputs and if my code is functioning correctly. And ... it does! I create and input layer with a single weight of 1.0 and bias of 0.0 andn no activation function. This will just feed forward any future input as itself. The hidden layer and output layer are created and random weights and biases generated.

- Day 8: In /testing/layers I now have forward propagation through all the layers. I am able to feed in a vector of inputs and return a vector of outputs. I learned that the function Softmax is a function applied to a vector of outputs rather than functioning like a standard activation function such as Sigmoid or ReLU. I had to restructure a few things in the program to move the Softmax function outside of the activation function enum and still get it to activate on my output layer.

- Day 9: In /testing/layers I created a branch and merged a feature for loading my training data as inputs from a csv file. I can pick the number of columns in order to load, and choose to skip a number of first columns for better data selection loading. This is used for loading my probabilities. Additionally, I added a function to shift my loaded input data by the mean value of the data column. Finally, I can loop through my full vector of inputs and get a complete vector of outputs from all my data. I think I am all setup for learning training and back propagation.

- Day 10: /testing/layers Today I started first steps into training and back propagation. I created a mean squared error function. takes my input data predictions and output predictions to calculate the loss of the network. Also, I made a function for getting the mean values of columns in a Vec<Vec<f64>> "table" and is now used in shift data, and mean_squared_error functions.
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
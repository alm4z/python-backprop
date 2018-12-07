# Vanilla backpropagation on Python

A simple Python script showing how the backpropagation algorithm works.
  - Numpy realization
  - Basic MLP neural network

# XOR Problem

  The backpropagation algorithm begins by comparing the 
actual value output by the forward propagation process to the expected
value and then moves backward through the network, slightly adjustingeach
of the weights in a direction that reduces the size of the error
by a small degree. Both forward and back propagation are re-run 
thousands of times on each input combination until the network can 
accurately predict the expected output of the possible inputs using 
forward propagation.

  For the xOr problem, 100% of possible data examples are available to use 
in the training process. We can therefore expect the trained network to be 
100% accurate in its predictions and there is no need to be concerned with 
issues such as bias and variance in the resulting model. [1]

# References
1. Well described XOR problem from [Jayesh Bapu Ahire](https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b)

# Contact
If you have any further questions or suggestions, please, do not hesitate to contact  by email at a.sadenov@gmail.com.

# License
MIT
# Deep_Neural_Network
The deep neural network architecture implemented here has a 5 layered structure (3 hidden layers) and it classifies the given input on a logical basis which is as follows :
The network will take in labelled data in the format as such:
The sum of the 10 neurons in each of the input layers of the dataset, if the sum is lesser than or equal to 5, the output of the neural network must be less than 0.5, and for the sum being anywhere from 5 to 10, the output must be greater than 0.5.
This may convieniant but keep in mind that there is ABSOLUTELY NO use of the fact of the 'sum' of the input layer neurons's values being used in the code.
The algorithm continues to learn in a supervised manner, after reviewing the labelled data and tweaking its paramters ever so sufficiently.
The idea of the 'sum' of the input layer neurons's values has merely been chosen to simulate the logicallities involved within the neural network and to metaphorically visualize the internal workings of the network.
In truth, a neural network that classifies the most irrelevant data based on a 'sum' factor isnt really that useful but all the aspects of a neural network are being encapsulated here.
The available functions for optimization and cost decreament using SGD aren't being employed, rather i have taken a more in-depth approach by hard-coding every change in every parameter in every iteration, as shown below.

![Screenshot 2024-03-17 135340](https://github.com/R2D2-08/Deep_Neural_Network/assets/155892663/76a1cf91-aa92-4be5-86e5-2f7ddf07aa42)

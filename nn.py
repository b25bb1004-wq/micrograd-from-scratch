"""Neuron, Layer and MLP: a small multi-layer perceptron built out of Value,
each neuron computing tanh(w . x + b)."""

import random

from engine import Value


# one neuron computes tanh(w . x + b)
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] #for nin xi, randomly pick wi
        self.b = Value(random.uniform(-1,1)) #for nin xi, randomly choose one bias

    def __call__(self, x):
        #w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b) #zip lines up wi with xi, and wi * xi for runs a for loop and multiplies them which is added up and then summed with b just once.
        out = act.tanh() # squash the weighted sum into (-1, 1)
        return out

    def parameters(self):
        return self.w + [self.b]

# a layer is a list of neurons that all see the same inputs
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)] #to generate weights and biases

    def __call__(self, x):
        outs = [n(x) for n in self.neurons] #to do the computation
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        #return [p for neuron in self.neurons for p in neuron.parameters()]
        params = []
        for neurons in self.neurons:
            ps = neurons.parameters()
            params.extend(ps)
        return params

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] #creates a list lets say nin = [3] and nouts = [4,4,1] so the number of features are 3 and number of neurons per layer is 4, 4, 1 respectively and there are 3 layers in total defined by len (nouts); hence, we get 3 weights per neuron times 4 in the first layer, which equals 12. Then 4 inputs to 4 neurons: we get 1 per neuron, hence 16 in total, and then 4 in 1 neuron, hence 4 outputs. Total is 32. And 9 biases
        #in a loop first 3,4 nin and nouts, then it goes on.

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x); #pass the input x to layers one by one, which does parallel computing across one layer
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

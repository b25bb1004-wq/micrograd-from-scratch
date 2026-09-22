"""Scalar autograd engine: a Value wraps a number and records how it was
computed, so backward() can walk that graph in reverse and fill in .grad
for every node via the chain rule."""

import math


# a scalar that remembers how it was computed, so gradients can flow backward through the graph
class Value:
    def __init__(self, data, _children=(), _op='', label = ''):
        self.data = data # the number this node holds
        self.grad = 0.0 # d(final output)/d(this node); stays 0 until backward() runs
        self._prev = set(_children) # the nodes this one was computed from
        self._backward = lambda: None # leaf nodes have nothing to propagate
        self._op = _op # operation that produced this node (used when drawing the graph)
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) #to allow quick mathematical calculations of object and other data type such as int or float
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(): # + passes the upstream grad to both inputs (local derivative = 1)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) #to allow quick mathematical calculations of object and other data type such as int or float
        out = Value(self.data * other.data, (self, other), '*')

        def _backward(): # chain rule: d(a*b)/da = b and d(a*b)/db = a, each times the upstream grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other): # other * self
        return self * other


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward(): # power rule: d(x^k)/dx = k * x^(k-1)
            self.grad += other * (self.data**(other - 1)) * out.grad

        out._backward = _backward

        return out


    def __rsub__(self, other): # other - self
        return other + (-self)

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1


    def tanh(self):
        x = self.data
        t = math.tanh(x) # same as (e^2x - 1)/(e^2x + 1), but doesn't overflow for large |x|
        out = Value(t, (self, ), 'tanh')

        def _backward(): # d(tanh x)/dx = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out  = Value(math.exp(x), (self, ), 'exp')

        def _backward(): # d(e^x)/dx = e^x
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def relu(self): # not used by this notebook's tanh-based Neuron, added for completeness
        out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward(): # gradient passes through where the input was positive, blocked otherwise
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v) #reach the leaves first, then start adding their nodes one by one to topo

        build_topo(self)

        self.grad = 1.0 # d(out)/d(out) = 1, the starting point of backprop
        for node in reversed(topo): # from the output back to the leaves
            node._backward()

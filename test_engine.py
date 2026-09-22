# tests for engine.py (Value) and nn.py (Neuron/Layer/MLP)
# run with: pytest
#
# a few of these are regression tests for actual bugs hit while debugging
# training - the comment on each one says which

import math
import random

import pytest

from engine import Value
from nn import MLP, Neuron, Layer


def grad(v, wrt):
    # zero every node in v's graph, run backward, hand back wrt's grad
    seen, stack = set(), [v]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        node.grad = 0.0
        stack.extend(node._prev)
    v.backward()
    return wrt.grad


# basic operators, both operand orders -----------------------------------

def test_add_and_radd():
    x = Value(2.0)
    assert (x + 3).data == 5.0
    assert (3 + x).data == 5.0
    assert grad(x + 3, x) == 1.0
    assert grad(3 + x, x) == 1.0

def test_sub_and_rsub():
    x = Value(2.0)
    assert (x - 3).data == -1.0
    assert (3 - x).data == 1.0
    assert grad(x - 3, x) == 1.0
    assert grad(3 - x, x) == -1.0  # d(c - x)/dx = -1

def test_mul_and_rmul():
    x = Value(2.0)
    assert (x * 3).data == 6.0
    assert (3 * x).data == 6.0
    assert grad(x * 3, x) == 3.0
    assert grad(3 * x, x) == 3.0

def test_truediv_and_rtruediv():
    # 1 / x used to crash, __rtruediv__ wasn't defined yet
    x = Value(2.0)
    assert (x / 2).data == 1.0
    assert (1 / x).data == 0.5
    assert grad(x / 2, x) == 0.5
    assert grad(1 / x, x) == pytest.approx(-1 / x.data**2)  # d(1/x)/dx = -1/x^2

def test_pow_and_neg():
    x = Value(3.0)
    assert (x**2).data == 9.0
    assert grad(x**2, x) == 6.0  # d(x^2)/dx = 2x
    assert (-x).data == -3.0

def test_division_by_zero_value_raises():
    # x / Value(0) goes through other**-1, i.e. 0**-1 -> ZeroDivisionError.
    # just documenting that this is what happens, not silently wrong
    with pytest.raises(ZeroDivisionError):
        Value(1.0) / Value(0.0)


# tanh / exp / relu --------------------------------------------------------

def test_tanh_matches_math_tanh():
    x = Value(0.7)
    assert x.tanh().data == pytest.approx(math.tanh(0.7))

def test_tanh_derivative():
    x = Value(0.5)
    out = x.tanh()
    assert grad(out, x) == pytest.approx(1 - math.tanh(0.5)**2)

def test_tanh_does_not_overflow_on_large_input():
    # tanh used to be built from math.exp(2*x), which overflows around
    # |x| > ~355 even though tanh(x) itself is just heading to +-1
    assert Value(400.0).tanh().data == pytest.approx(1.0)
    assert Value(-400.0).tanh().data == pytest.approx(-1.0)

def test_exp_still_overflows_on_very_large_input():
    # exp doesn't get the same fix - e^x genuinely can't fit in a float
    # once x is this big, so this is a documented limitation, not a bug
    with pytest.raises(OverflowError):
        Value(800.0).exp()

def test_relu():
    assert Value(-3.0).relu().data == 0
    assert Value(5.0).relu().data == 5.0
    assert grad(Value(-3.0).relu(), Value(-3.0)) == 0  # blocked on the negative side
    x = Value(5.0)
    assert grad(x.relu(), x) == 1  # passes straight through on the positive side


# graph shape edge cases ---------------------------------------------------

def test_value_used_twice_accumulates_gradient():
    # if _backward overwrote grad instead of +=, one of the two paths here
    # would just get clobbered
    x = Value(3.0)
    y = x + x  # dy/dx = 2
    y.backward()
    assert x.grad == 2.0

def test_diamond_graph():
    # a -> b, a -> c, b and c both feed d - checks grads get summed across
    # both paths back to a, not overwritten by whichever runs last
    a = Value(2.0)
    b = a * 3
    c = a * 5
    d = b + c
    d.backward()
    assert a.grad == 8.0  # d(3a + 5a)/da = 8

def test_backward_without_zeroing_accumulates_across_calls():
    # this is the actual bug hit while training: skip `p.grad = 0` between
    # steps and each backward() adds onto the previous grad instead of
    # replacing it, since every _backward uses +=
    x = Value(2.0)
    y = x * 3
    y.backward()
    first = x.grad
    y.backward()  # no zeroing in between
    assert x.grad == 2 * first


# Neuron / Layer / MLP -------------------------------------------------

def test_neuron_output_is_bounded_by_tanh():
    random.seed(0)
    neuron = Neuron(3)
    out = neuron([1.0, -2.0, 3.0])
    assert -1.0 <= out.data <= 1.0

def test_layer_output_count():
    random.seed(0)
    layer = Layer(3, 5)
    out = layer([1.0, 1.0, 1.0])
    assert len(out) == 5

def test_mlp_parameter_count():
    random.seed(0)
    mlp = MLP(3, [4, 4, 1])
    # 3->4: 4*(3+1)=16, 4->4: 4*(4+1)=20, 4->1: 1*(4+1)=5 -> 41 total
    assert len(mlp.parameters()) == 41

def test_mlp_output_is_a_single_value_for_one_output_neuron():
    random.seed(0)
    mlp = MLP(3, [4, 4, 1])
    out = mlp([1.0, 1.0, 1.0])
    assert isinstance(out, Value)


# gradient check against a plain numerical derivative -----------------------

def numerical_grad(mlp, xs, ys, param, eps=1e-6):
    def loss():
        return sum((mlp(x) - y)**2 for x, y in zip(xs, ys)).data
    orig = param.data
    param.data = orig + eps
    plus = loss()
    param.data = orig - eps
    minus = loss()
    param.data = orig
    return (plus - minus) / (2 * eps)

def test_mlp_analytic_gradient_matches_numerical_gradient():
    random.seed(0)
    mlp = MLP(3, [4, 4, 1])
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    for p in mlp.parameters():
        p.grad = 0
    loss = sum((mlp(x) - y)**2 for x, y in zip(xs, ys))
    loss.backward()

    # all 41 params would be overkill, a handful spread across layers is
    # enough to catch a broken chain rule anywhere in the network
    sample = mlp.parameters()[::7]
    for p in sample:
        assert p.grad == pytest.approx(numerical_grad(mlp, xs, ys, p), abs=1e-4)


# does it actually train --------------------------------------------------

def test_training_loop_decreases_loss():
    random.seed(0)
    mlp = MLP(3, [4, 4, 1])
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    losses = []
    for _ in range(50):
        pred = [mlp(x) for x in xs]
        loss = sum((yo - yg)**2 for yg, yo in zip(ys, pred))
        for p in mlp.parameters():
            p.grad = 0
        loss.backward()
        for p in mlp.parameters():
            p.data -= 0.05 * p.grad
        losses.append(loss.data)

    assert losses[-1] < losses[0]

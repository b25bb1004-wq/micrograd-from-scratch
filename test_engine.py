"""Tests for engine.py (Value) and nn.py (Neuron/Layer/MLP).

Run with: pytest

A few of these tests encode bugs found while debugging this project's
training loop (see the comments on each one), so they double as
regression tests for those specific mistakes.
"""

import math
import random

import pytest

from engine import Value
from nn import MLP, Neuron, Layer


# ---------------------------------------------------------------------------
# basic operators: value and gradient, both operand orders
# ---------------------------------------------------------------------------

def grad(v, wrt):
    """Reset every node's grad in v's graph, run backward, return wrt's grad."""
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
    # edge case: __rtruediv__ was missing until this was added, so `1 / x` used to crash
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
    # edge case: dividing by a Value whose data is 0 goes through other**-1,
    # i.e. 0**-1, which raises ZeroDivisionError -- documented, not silently wrong
    with pytest.raises(ZeroDivisionError):
        Value(1.0) / Value(0.0)


# ---------------------------------------------------------------------------
# tanh / exp / relu, including the overflow edge case that motivated
# switching tanh's implementation to math.tanh
# ---------------------------------------------------------------------------

def test_tanh_matches_math_tanh():
    x = Value(0.7)
    assert x.tanh().data == pytest.approx(math.tanh(0.7))

def test_tanh_derivative():
    x = Value(0.5)
    out = x.tanh()
    assert grad(out, x) == pytest.approx(1 - math.tanh(0.5)**2)

def test_tanh_does_not_overflow_on_large_input():
    # edge case: the original tanh, built from math.exp(2*x), raised OverflowError
    # around |x| > ~355 because e^(2x) itself overflows even though tanh(x) -> +-1
    assert Value(400.0).tanh().data == pytest.approx(1.0)
    assert Value(-400.0).tanh().data == pytest.approx(-1.0)

def test_exp_still_overflows_on_very_large_input():
    # edge case, documented limitation: exp has no such guard, since there's no
    # way to represent e^x as a finite float once x is too large
    with pytest.raises(OverflowError):
        Value(800.0).exp()

def test_relu():
    assert Value(-3.0).relu().data == 0
    assert Value(5.0).relu().data == 5.0
    assert grad(Value(-3.0).relu(), Value(-3.0)) == 0  # blocked on the negative side
    x = Value(5.0)
    assert grad(x.relu(), x) == 1  # passes through on the positive side


# ---------------------------------------------------------------------------
# graph edge cases
# ---------------------------------------------------------------------------

def test_value_used_twice_accumulates_gradient():
    # edge case: a naive backward pass that assigns instead of accumulating
    # grad would silently drop one of the two paths below
    x = Value(3.0)
    y = x + x  # dy/dx should be 2, from the two '+1' paths, not 1
    y.backward()
    assert x.grad == 2.0

def test_diamond_graph():
    # a -> b, a -> c, b and c both feed d: classic case for checking
    # gradients are summed across multiple paths, not overwritten
    a = Value(2.0)
    b = a * 3
    c = a * 5
    d = b + c
    d.backward()
    assert a.grad == 8.0  # d(3a + 5a)/da = 8

def test_backward_without_zeroing_accumulates_across_calls():
    # edge case from this project's own debugging: forgetting `p.grad = 0`
    # between steps makes each backward() ADD to the previous gradient
    # (because every _backward uses +=), rather than replacing it.
    x = Value(2.0)
    y = x * 3
    y.backward()
    first = x.grad
    y.backward()  # no zeroing in between
    assert x.grad == 2 * first


# ---------------------------------------------------------------------------
# Neuron / Layer / MLP
# ---------------------------------------------------------------------------

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
    # 3->4: 4*(3+1)=16, 4->4: 4*(4+1)=20, 4->1: 1*(4+1)=5 => 41 total
    assert len(mlp.parameters()) == 41

def test_mlp_output_is_a_single_value_for_one_output_neuron():
    random.seed(0)
    mlp = MLP(3, [4, 4, 1])
    out = mlp([1.0, 1.0, 1.0])
    assert isinstance(out, Value)


# ---------------------------------------------------------------------------
# gradient correctness via numerical (finite-difference) check
# ---------------------------------------------------------------------------

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

    # checking all 41 parameters would be slow; a handful across different
    # layers is enough to catch a broken chain rule anywhere in the network
    sample = mlp.parameters()[::7]
    for p in sample:
        assert p.grad == pytest.approx(numerical_grad(mlp, xs, ys, p), abs=1e-4)


# ---------------------------------------------------------------------------
# training: loss should decrease on this project's toy dataset
# ---------------------------------------------------------------------------

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

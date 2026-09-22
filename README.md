# micrograd from scratch

A scalar autograd engine and a small multi-layer perceptron, built from
scratch in Python, following Andrej Karpathy's ["The spelled-out intro to
neural networks and backpropagation: building
micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0) lecture.

`Value` wraps a number and records how it was computed, so that calling
`.backward()` on the final output walks the graph in reverse and fills in
`.grad` for every node via the chain rule — the same idea PyTorch's autograd
is built on, at a scale you can read top to bottom in one sitting.

## Layout

| File | Contents |
|---|---|
| [`engine.py`](engine.py) | `Value`: the scalar autograd engine (`+`, `-`, `*`, `/`, `**`, `tanh`, `exp`, `relu`, `backward`) |
| [`nn.py`](nn.py) | `Neuron`, `Layer`, `MLP`, built on top of `Value` |
| [`micrograd_from_scratch.ipynb`](micrograd_from_scratch.ipynb) | The walkthrough: derivatives, `Value`, a hand-built neuron, the graph visualizer, and training a small MLP on a toy dataset |
| [`test_engine.py`](test_engine.py) | Correctness and edge-case tests for `engine.py` / `nn.py` |

## Setup

```bash
pip install -r requirements.txt
```

`graphviz` (the Python package) also needs the `dot` binary installed
separately to actually render diagrams (`brew install graphviz` on macOS).
Everything else in the notebook works without it.

## Running the notebook

Open `micrograd_from_scratch.ipynb` in Jupyter and run the cells top to
bottom. A few things worth knowing before stepping through it manually
instead of using the training-loop cell:

- **Create the network once.** `n = MLP(3, [4, 4, 1])` should only run at
  the start. Re-running it throws away whatever the network has learned
  and starts over from new random weights — this is the single most
  common reason the loss looks like it's "not decreasing."
- **Each step needs a fresh forward pass.** `backward()` differentiates
  through *whatever graph was last built*. Update the parameters, then
  recompute `ypred` before computing the next loss — otherwise you're
  calling `backward()` on a stale graph.
- **Zero the gradients before every `backward()` call.** Every `_backward`
  uses `+=`, so gradients accumulate across calls by design (it's what
  lets a value used twice in an expression get contributions from both
  paths). Forgetting to zero them between training steps makes each step
  use the sum of every gradient computed so far, not just the current
  one.
- **Learning rate matters more than it looks.** At `lr = 0.05` the loss
  can bounce for the first several dozen steps before settling — that's
  normal overshoot, not a bug. `lr = 0.01` is slower but smoother.

## Tests

```bash
pytest
```

Covers the basic operators in both operand orders (`x - 1` and `1 - x`),
gradient correctness (including a numerical finite-difference check
against the analytic gradients produced by `backward()`), graph edge
cases (a value used twice, diamond-shaped graphs), `relu`, and two edge
cases found while debugging this project: `tanh` no longer overflowing
on large inputs, and what happens if you forget to zero gradients
between training steps.

## Known limitations

- `exp()` raises `OverflowError` for inputs above roughly 710, since
  `e^x` no longer fits in a float at that point. `tanh()` doesn't have
  this problem even for very large inputs, since it's implemented with
  `math.tanh` directly rather than via `exp`.
- `Value.__pow__` only supports `int`/`float` exponents (i.e. `x**2` and
  `x**-1`, but not `x**y` for another `Value` `y`).

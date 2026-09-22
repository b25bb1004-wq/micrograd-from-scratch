# micrograd from scratch

A small autograd engine and a tiny MLP, built from scratch in Python while following Andrej Karpathy's "Building micrograd" lecture.

`Value` wraps a number and keeps track of how it was computed, so calling `.backward()` on the final output walks back through that graph and fills in `.grad` for every node using the chain rule. It's the same basic idea behind PyTorch's autograd, just small enough to read end to end.

## Contents

- `engine.py` - the `Value` class (`+`, `-`, `*`, `/`, `**`, `tanh`, `exp`, `relu`, `backward`)
- `nn.py` - `Neuron`, `Layer`, and `MLP`, built on top of `Value`
- `micrograd_from_scratch.ipynb` - the walkthrough: derivatives, `Value`, a hand-built neuron, the graph visualizer, and training a small MLP on a toy dataset
- `test_engine.py` - tests for `engine.py` and `nn.py`

## Setup

```
pip install -r requirements.txt
```

Graphviz also needs the `dot` binary installed on your system to actually render diagrams (`brew install graphviz` on Mac). Everything else in the notebook works without it.

## Running the notebook

Open `micrograd_from_scratch.ipynb` and run the cells top to bottom. A few things to know if you're stepping through it manually instead of using the training loop cell:

- Create the network once. `n = MLP(3, [4, 4, 1])` should only run at the start - re-running it resets every weight to random again, which looks exactly like the loss isn't learning.
- `backward()` only knows about whatever graph you last built, so you need a fresh forward pass before every backward call, not just the loss and update cells.
- Zero the gradients before every backward call. They accumulate with `+=` instead of overwriting, so skipping this makes each step use the sum of every gradient computed so far.
- `lr = 0.05` bounces for the first several dozen steps before it settles down. That's normal overshoot, not a bug.

## Tests

```
pytest
```

Covers the operators in both operand orders (`x - 1` and `1 - x`), gradient correctness against a numerical finite-difference check, a couple of graph edge cases (a value reused twice, a diamond-shaped graph), and two regression tests for real bugs hit while debugging this project - `tanh` overflowing on large inputs, and forgetting to zero gradients between training steps.

## Known limitations

- `exp()` raises `OverflowError` for inputs above roughly 700, since `e^x` no longer fits in a float at that point. `tanh()` doesn't have this issue, since it's implemented with `math.tanh` directly rather than through `exp`.
- `Value.__pow__` only supports `int`/`float` exponents, so `x**2` works but `x**y` for another `Value` `y` doesn't.

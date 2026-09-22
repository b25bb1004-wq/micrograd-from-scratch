# micrograd from scratch

following along with karpathy's "building micrograd" video, implementing a tiny autograd engine and a small MLP by hand.

`Value` wraps a number and remembers how it was computed, so `.backward()` can walk back through the graph and work out the gradient of everything with respect to the final output. basically how pytorch's autograd works, just at a scale you can actually read top to bottom.

## what's here

- `engine.py` - the `Value` class (+, -, *, /, **, tanh, exp, relu, backward)
- `nn.py` - `Neuron`, `Layer`, `MLP`, built on top of `Value`
- `micrograd_from_scratch.ipynb` - the actual walkthrough: derivatives -> Value -> a neuron by hand -> the graph viz -> training a tiny MLP
- `test_engine.py` - tests for the above

## running it

```
pip install -r requirements.txt
```

graphviz also needs the `dot` binary installed on your system to actually draw the diagrams (`brew install graphviz` on mac). everything else works without it.

## things that tripped me up

- only create the MLP once (`n = MLP(3, [4, 4, 1])`). re-running that cell resets every weight back to random, which looks exactly like the loss "not learning" if you're stepping through cells by hand instead of using the loop
- backward() only knows about whatever graph you last built, so you need a fresh forward pass before every backward() call, not just re-run the loss/update cells
- zero the grads before every backward() call, they accumulate (+=) instead of overwriting. forgot this once and it made training look way better than it actually was
- lr=0.05 bounces around for the first several dozen steps before it actually settles down, that's normal, not a bug

## tests

```
pytest
```

## known limitations

- `exp()` overflows for inputs much above ~700, since e^x just doesn't fit in a float at that point. tanh doesn't have this problem since it's implemented with math.tanh directly instead of going through exp
- `**` only works with int/float exponents, so `x**2` is fine but `x**y` where y is also a Value isn't

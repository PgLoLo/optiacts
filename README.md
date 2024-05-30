# OptiActs

Point-wise nonlinearity functions that saves their output instead of input.


# Installation
```bash
pip install git+https://github.com/PgLoLo/optiacts
``` 


# How it works
Many nonlinearity functions save their input for the backward pass in order to perform automatic differentiation:

```
forward: X -> Y = f(X)
save:    X
backward: dL / dX = dL / dY * f'(X)   
```

Our implementation, instead, saves the output tensor for backward pass:
```
forward: X -> Y = f(X)
save:    Y
backward: dL / dX = dL / dY * f'(f^-1(Y))
```
In the case when there is another layer after the nonlinearity that would save its input tensor, only one tensor `Y` would be saved, compared to two tensors `X` and `Y` in the standard case.

For GELU and SiLU nonlinearities, `f` is not invertible, thus we save additional information (only one bit per element is required to make `f` invertible).


# Examples
We implemented drop-in replacements for `torch.nn.GELU` and `torch.nn.SiLU` layers and for `torch.nn.functional.gelu` and `torch.nn.functional.silu` functions:

```python
import optiacts
optiacts.GELU()  # torch.nn.Module
optiacts.SiLU()  # torch.nn.Module
optiacts.gelu    # function
optiacts.silu    # function
```

You can use them inside your `torch.nn.Module`-s as usual, or replace in already constructed networks. Here is, for example, how to create a Hugging Face BERT model and replace activation functions there:
```python
import optiacts
from transformers import BertConfig, BertModel

model = BertModel(BertConfig())
for layer in model.encoder.layer:
    layer.intermediate.intermediate_act_fn = optiacts.GELU()
```
The exact way to replace all nonlinearities in a given model, of course, depends on the architecture and implementation of the given model.


# How Much Memory is Saved?
The exact amount of memory saved, of course, varies for different use cases

For the Mistral-7b model, our method achieves a 15.7% reduction in memory for all saved activations (see [Mistral](./examples/Mistral.ipynb) example).
For the BERT-standard model, it achieves 23.1% reduction (see [BERT](./examples/BERT.ipynb) example).


# Is It Slower?
Our method is indeed a little bit slower right now, but not by much: in our experiments registered slowdown is less than 1%.

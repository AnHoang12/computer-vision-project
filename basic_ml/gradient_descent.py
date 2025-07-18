import autograd.numpy as np
from autograd import grad

def f(x):
    return x**2 + 2*x + 5

grad_f = grad(f)

x = -10.0
lr = 0.1
epochs = 100

for i in range(epochs):
    gradient = grad_f(x)
    x = x - lr * gradient
    print(f"Epoch {i+1}: x = {x:6f}, f(x) = {f(x):6f}")

print (f"Min = {f(x)} where x = {x}")
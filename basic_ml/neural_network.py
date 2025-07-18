import numpy as np

"""Build a neural network to define XOR function"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array(
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
)

Y = np.array(
    [[0],
     [0],
     [0],
     [1]]
)

input_layer = 2
hidden_layer = 4
output = 1

np.random.seed(42)
w1 = np.random.uniform(size=(input_layer, hidden_layer)) # (2, 4)
w2 = np.random.uniform(size=(hidden_layer, output)) # (4, 1)

epochs = 15000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, w1)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, w2)
    predict_output = sigmoid(final_input)

    # Loss
    error = Y - predict_output
    mse = np.mean(error**2)

    # Backpropagation
    d_predict_output = error * sigmoid_derivative(predict_output)
    error_hidden_layer = d_predict_output.dot(w2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Update parameters
    w2 += hidden_output.T.dot(d_predict_output) * learning_rate #
    w1 += X.T.dot(d_hidden_layer) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss = {mse:.6f}")

print("Output")
print(predict_output.round(3))  

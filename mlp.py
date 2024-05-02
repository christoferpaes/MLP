import numpy as np
import plotly.graph_objects as go
import warnings

# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases for the hidden layer
        self.weights_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)

        # Initialize weights and biases for the output layer
        self.weights_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        # Forward pass
        hidden_activation = self.sigmoid(np.dot(inputs, self.weights_hidden) + self.bias_hidden)
        output_activation = self.sigmoid(np.dot(hidden_activation, self.weights_output) + self.bias_output)
        return output_activation

    def train(self, inputs, targets, epochs=9999):
        for _ in range(epochs):
            # Forward pass
            hidden_activation = self.sigmoid(np.dot(inputs, self.weights_hidden) + self.bias_hidden)
            output_activation = self.sigmoid(np.dot(hidden_activation, self.weights_output) + self.bias_output)

            # Backpropagation
            output_error = targets - output_activation
            output_delta = output_error * output_activation * (1 - output_activation)

            hidden_error = np.dot(output_delta, self.weights_output.T)
            hidden_delta = hidden_error * hidden_activation * (1 - hidden_activation)

            # Update weights and biases
            self.weights_output += self.learning_rate * np.dot(hidden_activation.T, output_delta)
            self.bias_output += self.learning_rate * np.sum(output_delta, axis=0)

            self.weights_hidden += self.learning_rate * np.dot(inputs.T, hidden_delta)
            self.bias_hidden += self.learning_rate * np.sum(hidden_delta, axis=0)

    def plot_decision_boundary(self, inputs, targets):
        x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
        y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        # Predict the output for each point on the mesh grid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Create 3D scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(x=inputs[:, 0], y=inputs[:, 1], z=targets.flatten(),
                         mode='markers', marker=dict(color=targets.flatten(), size=5),
                         name='Data Points'),
            go.Surface(x=xx, y=yy, z=Z, opacity=0.6,
                       colorscale='Viridis', showscale=False,
                       name='Decision Boundary')
        ])

        fig.update_layout(scene=dict(
            xaxis_title='Input 1',
            yaxis_title='Input 2',
            zaxis_title='Output',
        ))

        fig.show()

# Define XOR logic gate truth table
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])  # XOR truth table with shape (4, 1)

# Create and train the MLP
mlp = MLP(input_size=2, hidden_size=4, output_size=1)
mlp.train(inputs, targets)

# Plot decision boundary in 3D

# Function to get user input and predict
def guess_xor_function(mlp):
    while True:
        try:
            input1 = int(input("Enter first input (0 or 1): "))
            input2 = int(input("Enter second input (0 or 1): "))
            guess = np.array([[input1, input2]])
            prediction = mlp.predict(guess)
            roundedPrediction = round(float(prediction))
            print(f"Predicted output: {roundedPrediction}")
            mlp.plot_decision_boundary(inputs, targets)

            break
        except ValueError:
            print("Invalid input. Please enter 0 or 1.")

# Run the prediction function
guess_xor_function(mlp)

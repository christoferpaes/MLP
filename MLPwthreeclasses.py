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
    def guess(self):
        while True:
            try:
                input1 = float(input("Enter first input: "))
                input2 = float(input("Enter second input: "))
                guess = np.array([[input1, input2]])
                prediction = self.predict(guess)
                class_names = ['High-Spending', 'Medium-Spending', 'Low-Spending']
                predicted_class_index = np.argmax(prediction)
                predicted_class = class_names[predicted_class_index]
                print(f"Predicted class: {predicted_class}")
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def predict(self, inputs):
        # Forward pass
        hidden_activation = self.sigmoid(np.dot(inputs, self.weights_hidden) + self.bias_hidden)
        output_activation = self.softmax(np.dot(hidden_activation, self.weights_output) + self.bias_output)
        return output_activation

    def train(self, inputs, targets, epochs=9999):
        for _ in range(epochs):
            # Forward pass
            hidden_activation = self.sigmoid(np.dot(inputs, self.weights_hidden) + self.bias_hidden)
            output_activation = self.softmax(np.dot(hidden_activation, self.weights_output) + self.bias_output)

            # Backpropagation
            output_error = targets - output_activation
            output_delta = output_error

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
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        # Create a 3D scatter plot for the data points
        fig = go.Figure()

        # Separate points for each class
        for class_idx in range(self.output_size):
            class_inputs = inputs[np.argmax(targets, axis=1) == class_idx]
            fig.add_trace(go.Scatter3d(x=class_inputs[:, 0], y=class_inputs[:, 1], z=np.zeros_like(class_inputs[:, 0]) + class_idx,
                                        mode='markers', marker=dict(size=5, color=class_idx),
                                        name=f'Class {class_idx}'))

        # Add the decision boundary to the plot
        fig.add_trace(go.Surface(x=xx, y=yy, z=Z, opacity=0.6,
                                  colorscale='Viridis', showscale=False,
                                  name='Decision Boundary'))

        # Set plot layout
        fig.update_layout(scene=dict(
            xaxis=dict(title='Input 1', range=[x_min, x_max]),
            yaxis=dict(title='Input 2', range=[y_min, y_max]),
            zaxis=dict(title='Output')
        ))

        # Show the plot
        fig.show()



# Sample inputs and targets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.2], [0.8, 0.9]])
targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Create and train the MLP
mlp = MLP(input_size=2, hidden_size=4, output_size=3)
mlp.train(inputs, targets)

# Plot decision boundary
mlp.plot_decision_boundary(inputs, targets)
mlp.guess()

# MNIST Neural Network Classifier

A simple feedforward neural network implementation from scratch using NumPy for handwritten digit recognition on the MNIST dataset.

## Project Overview

This project implements a 3-layer neural network capable of classifying handwritten digits (0-9).

**Key Features:**

- Simple 3-layer feedforward architecture
- Built-in training visualization and testing
- Model saving/loading functionality

## Architecture

The neural network consists of three layers:

- **Input Layer**: 784 neurons (28×28 pixel values)
- **Hidden Layer**: Configurable size (default: 64 neurons) with ReLU activation
- **Output Layer**: 10 neurons (digit classes 0-9) with Softmax activation

```
Input (784) → Hidden (64) → Output (10)
    ↓           ↓            ↓
  Pixels     ReLU(x)    Softmax(x)
```

## 📦 Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   uv sync
   ```

## 🚀 Quick Start

Simply run the main script:

```bash
uv run main.py
```

**What happens:**

- If a trained model (`simple-nn.npz`) exists, it loads and tests it
- Otherwise, trains a new model for 800 iterations
- Displays predictions on random test images with visualization

## Performance

With my current settings, the following performance is achieved:

- **Training Accuracy**: ~91%+
- **Test Accuracy**: ~92%+

## 🔧 Configuration

You can customize the network in the `main()` function:

```python
nn = SimpleNN(
    hidden_layer_size=64,    # Number of hidden neurons
    learning_rate=0.05,      # Learning rate
    kaiming=True            # Use Kaiming initialization
)
```

## 📁 Project Structure

```
├── main.py              # Main implementation
├── pyproject.toml       # Project dependencies
├── README.md           # This file
├── simple-nn.npz       # Saved model (generated after training)
└── MNIST data files    # Downloaded automatically
```

## 🧮 Mathematical Foundation

### Forward Propagation

For a batch of $M$ examples:

**Layer 1 (Hidden):**  
$$Z^{[1]} = W^{[1]}X + b^{[1]}$$  
$$A^{[1]} = \text{ReLU}(Z^{[1]})$$

**Layer 2 (Output):**  
$$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$$  
$$A^{[2]} = \text{Softmax}(Z^{[2]})$$

### Backpropagation

**Output Layer Gradients:**  
$$dZ^{[2]} = A^{[2]} - Y$$  
$$dW^{[2]} = \frac{1}{M} dZ^{[2]} \cdot (A^{[1]})^T$$  
$$db^{[2]} = \frac{1}{M} \sum dZ^{[2]}$$

**Hidden Layer Gradients:**  
$$dZ^{[1]} = (W^{[2]})^T \cdot dZ^{[2]} \odot \text{ReLU}'(Z^{[1]})$$  
$$dW^{[1]} = \frac{1}{M} dZ^{[1]} \cdot X^T$$  
$$db^{[1]} = \frac{1}{M} \sum dZ^{[1]}$$

Where:

- $M$ = batch size (number of examples)
- $\odot$ = element-wise multiplication
- $\text{ReLU}'(x) = \mathbf{1}_{x > 0}$ (derivative of ReLU)

## 🎨 Activation Functions

- **ReLU**: $f(x) = \max(0, x)$ - Fast, prevents vanishing gradients
- **Softmax**: $f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ - Probability distribution over classes

## 🔍 Key Implementation Details

- **Weight Initialization**: [Kaiming initialization](https://www.numberanalytics.com/blog/ultimate-guide-he-initialization-neural-networks) for optimal convergence
- **Loss Function**: choose MSE (default) or Categorical cross-entropy
- **Optimizer**: Standard gradient descent
- **Numerical Stability**: Softmax overflow prevention

## 🛠️ Extending the Code

Want to experiment? Try:

- Adding more hidden layers, changing number of neurons in the layer
- Implementing different optimizers (Adam, RMSprop)
- Adding dropout for regularization
- Experimenting with different activation functions

## 📚 Learning Resources

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Kaiming Initialization Paper](https://arxiv.org/abs/1502.01852)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## 📜 License

This project is for educational purposes. Feel free to use and modify as needed.

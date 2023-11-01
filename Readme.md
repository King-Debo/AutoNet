# AutoNet: A Program for Automatic Design and Optimization of Neural Networks

AutoNet is a program that can automatically design and optimize neural networks for different tasks and domains, such as computer vision, natural language processing, speech recognition, etc. The program uses meta-learning, evolutionary algorithms, and neural architecture search to find the best network architecture and hyperparameters for each task and domain.

## Table of Contents

- Installation
- Usage
- Requirements
- Modules
  - User Interface
  - Data Loader
  - Meta-Learner
  - Evolutionary Algorithm
  - Neural Architecture Search
  - Hyperparameter Optimization
  - Trainer
  - Evaluator
  - Saver
  - Loader
- Credits
- License
- Code of Conduct

## Installation

To install AutoNet, you need to have Python 3.6 or higher and PyTorch 1.8 or higher installed on your system. You also need to install the following libraries and modules:

- torchmeta
- deap
- autoPyTorch
- optuna
- pytorch_lightning
- torchmetrics
- tkinter
- torchvision

You can install them using pip or conda commands. For example:

```bash
pip install torchmeta deap autoPyTorch optuna pytorch_lightning torchmetrics tkinter torchvision

## Requirements

AutoNet requires the following data and resources to work properly:

- A dataset that contains input data and labels for your task and domain. The dataset should be in a pandas dataframe or a numpy array format.
- A meta-learning dataset that matches your task and domain of interest. The meta-learning dataset should be one of the following: Omniglot, MiniImagenet, CIFARFS, or FC100.
- A meta-learning model that matches your task and domain of interest. The meta-learning model should be one of the following: MetaConv2d, MetaLinear, MetaSequential, or MetaModule.
- A validation set or a test set to measure the performance of the final neural network using appropriate metrics.
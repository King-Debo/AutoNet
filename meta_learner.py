# Creating a meta-learner that learns from previous tasks and domains and transfers the knowledge to new tasks and domains

# Importing the meta-learning algorithms, datasets, and models from torchmeta
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear, MetaConv2d
from torchmeta.datasets import Omniglot, MiniImagenet, CIFARFS, FC100
from torchmeta.algorithms import ModelAgnosticMetaLearning, Reptile, MetaSGD

# Defining a function to get a meta-learning dataset based on the user input
def get_meta_dataset(task, domain):
    # Choosing a meta-learning dataset that matches the task and domain of interest
    if task == "classification" and domain == "characters":
        meta_dataset = Omniglot("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
    elif task == "classification" and domain == "images":
        meta_dataset = MiniImagenet("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
    elif task == "classification" and domain == "objects":
        meta_dataset = CIFARFS("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
    elif task == "classification" and domain == "scenes":
        meta_dataset = FC100("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
    else:
        meta_dataset = None # No meta-learning dataset for other tasks and domains
    
    # Returning the meta-learning dataset
    return meta_dataset

# Defining a function to get a meta-learning model based on the user input
def get_meta_model(task, domain):
    # Choosing a meta-learning model that matches the task and domain of interest
    if task == "classification" and domain == "characters":
        meta_model = MetaSequential(
            MetaConv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            MetaLinear(64, 5)
        )
    elif task == "classification" and domain == "images":
        meta_model = MetaSequential(
            MetaConv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            MetaLinear(128 * 5 * 5 , 5)
        )
    elif task == "classification" and domain == "objects":
        meta_model = MetaSequential(
            MetaConv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(64, 96 , kernel_size=3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(96 , 128 , kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(8),
            nn.Flatten(),
            MetaLinear(128 , 5)
        )
     # Continuing the MetaSequential model for the 'scenes' domain
    elif task == "classification" and domain == "scenes":
        meta_model = MetaSequential(
            MetaConv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MetaConv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            MetaLinear(256 * 6 * 6, 5)
        )

# Adding a softmax layer to output probabilities for five classes
meta_model.add_module("softmax", nn.Softmax(dim=1))

# Adding a log-likelihood loss function to measure the performance of the model
meta_model.add_module("loss", nn.NLLLoss())


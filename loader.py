# Creating the loader module

# Importing the methods from PyTorch
import torch

# Defining a function to load the saved neural network from a file or a database
def load_final_network(architecture, file_name):
    # Creating an instance of the architecture (a MetaSequential model)
    model = architecture.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Loading the model state dictionary using torch.load
    model.load_state_dict(torch.load(file_name))
    
    # Printing a confirmation message to the console
    print(f"Model loaded from {file_name}")
    
    # Returning the loaded model
    return model

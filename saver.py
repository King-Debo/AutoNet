# Creating the saver module

# Importing the methods from PyTorch
import torch

# Defining a function to save the final neural network in a file or a database
def save_final_network(model, file_name):
    # Saving the model state dictionary using torch.save
    torch.save(model.state_dict(), file_name)
    
    # Printing a confirmation message to the console
    print(f"Model saved as {file_name}")

# Creating the trainer module

# Importing the methods from PyTorch Lightning
import pytorch_lightning as pl

# Defining a function to train the final neural network using PyTorch Lightning
def train_final_network(architecture, hyperparameters):
    # Converting the architecture (a MetaSequential model) to a PyTorch model
    model = architecture.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Creating a data module object using PyTorch Lightning
    data_module = pl.LightningDataModule.from_data_loaders(data_loader) # Using the data loader from the previous module
    
    # Creating a model object using PyTorch Lightning
    model = pl.LightningModule(model) # Wrapping the model with PyTorch Lightning
    model.configure_optimizers = lambda: torch.optim.Adam(model.parameters(), **hyperparameters) # Defining the optimizer with the given hyperparameters
    model.criterion = nn.CrossEntropyLoss() # Defining the loss function as cross entropy loss
    model.training_step = lambda batch, batch_idx: model.criterion(model(batch[0]), batch[1]) # Defining the training step as calculating the loss between the model output and the label
    model.validation_step = lambda batch, batch_idx: model.criterion(model(batch[0]), batch[1]) # Defining the validation step as calculating the loss between the model output and the label
    
    # Creating a trainer object using PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=100, # Setting the maximum number of epochs
        gpus=1 if torch.cuda.is_available() else None, # Using GPU if available
        logger=True, # Enabling logging
        checkpoint_callback=True # Enabling checkpointing
    )
    
    # Fitting the model on the data using the trainer
    trainer.fit(model, data_module)
    
    # Returning the trained model and the trainer
    return model, trainer

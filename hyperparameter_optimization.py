# Creating the hyperparameter optimization module

# Importing the methods from Optuna
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Defining a function to optimize the hyperparameters of a given neural network architecture using Optuna
def hyperparameter_optimization(architecture):
    # Converting the architecture (a MetaSequential model) to a PyTorch model
    model = architecture.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Defining an objective function for Optuna
    def objective(trial):
        # Suggesting values for the hyperparameters using Optuna
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 256, log=True)
        
        # Creating a trainer object using PyTorch Lightning
        trainer = pl.Trainer(
            max_epochs=10, # Setting the maximum number of epochs
            gpus=1 if torch.cuda.is_available() else None, # Using GPU if available
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")], # Using pruning callback to stop unpromising trials
            logger=False, # Disabling logging
            checkpoint_callback=False # Disabling checkpointing
        )
        
        # Creating a data module object using PyTorch Lightning
        data_module = pl.LightningDataModule.from_data_loaders(data_loader) # Using the data loader from the previous module
        
        # Creating a model object using PyTorch Lightning
        model = pl.LightningModule(model) # Wrapping the model with PyTorch Lightning
        model.configure_optimizers = lambda: torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Defining the optimizer with the suggested hyperparameters
        model.criterion = nn.CrossEntropyLoss() # Defining the loss function as cross entropy loss
        model.training_step = lambda batch, batch_idx: model.criterion(model(batch[0]), batch[1]) # Defining the training step as calculating the loss between the model output and the label
        model.validation_step = lambda batch, batch_idx: model.criterion(model(batch[0]), batch[1]) # Defining the validation step as calculating the loss between the model output and the label
        
        # Fitting the model on the data using the trainer
        trainer.fit(model, data_module)
        
        # Returning the validation loss as the objective value
        return trainer.callback_metrics["val_loss"].item()
    
    # Creating a study object using Optuna
    study = optuna.create_study(direction="minimize") # Minimizing the objective value
    
    # Optimizing the study using Optuna
    study.optimize(objective, n_trials=100) # Running 100 trials
    
    # Returning the best trial and its hyperparameters and objective value
    return study.best_trial

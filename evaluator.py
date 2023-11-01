# Creating the evaluator module

# Importing the methods from PyTorch Metrics
import torchmetrics

# Defining a function to test the performance of the final neural network using PyTorch Metrics
def test_final_network(model, trainer):
    # Creating a data module object using PyTorch Lightning
    data_module = pl.LightningDataModule.from_data_loaders(data_loader) # Using the data loader from the previous module
    
    # Creating a metrics object using PyTorch Metrics
    metrics = torchmetrics.MetricCollection([
        torchmetrics.Accuracy(), # Accuracy metric
        torchmetrics.Precision(), # Precision metric
        torchmetrics.Recall(), # Recall metric
        torchmetrics.F1(), # F1-score metric
    ])
    
    # Testing the model on the data using the trainer and the metrics
    trainer.test(model, data_module, verbose=True, ckpt_path="best", callbacks=[metrics])
    
    # Printing the test results to the console
    print(f"Test results: {trainer.callback_metrics}")

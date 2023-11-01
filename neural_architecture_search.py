# Creating the neural architecture search module

# Importing the methods from Auto-PyTorch
from autoPyTorch import AutoNetClassification
from autoPyTorch.data_management.data_manager import DataManager

# Defining a function to evaluate the accuracy of a given neural network architecture using Auto-PyTorch
def neural_architecture_search(architecture):
    # Converting the architecture (a MetaSequential model) to a PyTorch model
    model = architecture.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Creating a data manager object using the user input data
    data_manager = DataManager()
    data_manager.generate_train_valid_splits(input_data, 0.2) # Splitting the input data into 80% train and 20% valid
    
    # Creating an AutoNetClassification object using the default configuration
    autonet = AutoNetClassification("tiny_cs", max_runtime=300, min_budget=30, max_budget=90)
    
    # Fitting the AutoNetClassification object on the train data using the given model as the network backbone
    autonet.fit(data_manager, network_backbone=model)
    
    # Predicting the labels of the valid data using the fitted AutoNetClassification object
    predictions = autonet.predict(data_manager.get_valid_data())
    
    # Computing the accuracy of the predictions using sklearn.metrics.accuracy_score
    accuracy = sklearn.metrics.accuracy_score(data_manager.get_valid_labels(), predictions)
    
    # Returning the accuracy
    return accuracy

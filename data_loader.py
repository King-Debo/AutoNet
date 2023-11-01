# Creating a data loader that preprocesses and transforms the input data into a suitable format for the neural network

# Defining a custom dataset class that inherits from torch.utils.data.Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_type, input_data):
        self.input_type = input_type # The type of input data, such as images, text, audio, etc.
        self.input_data = input_data # The input data in a pandas dataframe or a numpy array
        
        # Defining different transformations for different types of input data using torchvision.transforms
        if self.input_type == "images":
            self.transform = transforms.Compose([
                transforms.ToPILImage(), # Converting the images to PIL format
                transforms.Resize((224, 224)), # Resizing the images to 224 x 224 pixels
                transforms.ToTensor(), # Converting the images to tensors
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalizing the images using ImageNet statistics
            ])
        elif self.input_type == "text":
            self.transform = transforms.Compose([
                transforms.ToTensor(), # Converting the text to tensors
                transforms.Lambda(lambda x: x.long()) # Converting the text to long tensors
            ])
        elif self.input_type == "audio":
            self.transform = transforms.Compose([
                transforms.ToTensor(), # Converting the audio to tensors
                transforms.Lambda(lambda x: x.float()) # Converting the audio to float tensors
            ])
        else:
            self.transform = None # No transformation for other types of input data
    
    def __len__(self):
        return len(self.input_data) # Returning the length of the input data
    
    def __getitem__(self, index):
        # Getting the input data and the label at the given index
        data = self.input_data[index]
        label = data[-1] # Assuming the label is the last column of the input data
        
        # Applying the transformation to the input data if applicable
        if self.transform is not None:
            data = self.transform(data)
        
        # Returning the input data and the label as a tuple
        return (data, label)

# Creating an instance of the custom dataset class using the user input
dataset = CustomDataset(input_type, input_data)

# Creating a data loader using torch.utils.data.DataLoader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

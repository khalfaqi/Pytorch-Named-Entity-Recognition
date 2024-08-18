import torch
from preprocessing import NERDataset
from torch.utils.data import DataLoader
from model_ner import NERModel
import config
from preprocessing import labels_to_ids

# Define the evaluation function
def evaluate(model, data_test):
    test_dataset = NERDataset(data_test)  # Create a dataset for testing
    test_dataloader = DataLoader(test_dataset, batch_size=1)  # Create a data loader for testing

    model.eval()  # Set model to evaluation mode

    total_acc_test = 0.0  # Initialize accumulator for test accuracy

    with torch.no_grad():
        # Testing loop over batches
        for test_data, test_label in test_dataloader:
            input_ids = test_data['input_ids'].squeeze(1)  # Get input IDs and move to device
            attention_mask = test_data['attention_mask'].squeeze(1)  # Get attention mask and move to device
            labels = test_label  # Get labels and move to device

            outputs = model(input_ids, attention_mask, labels)  # Forward pass
            loss = outputs[0]
            logits = outputs[1]

            logits_clean = logits[labels != -100]  # Clean logits by removing ignored tokens
            labels_clean = labels[labels != -100]  # Clean labels by removing ignored tokens
            predictions = logits_clean.argmax(dim=1)  # Get predictions
            acc = (predictions == labels_clean).float().mean()  # Calculate accuracy
            total_acc_test += acc  # Accumulate accuracy

    test_accuracy = total_acc_test / len(test_dataloader)  # Calculate overall test accuracy
    print(f'Test Accuracy: {test_accuracy:.4f}')  # Print test accuracy

# Initialize and load the model
num_labels = len(labels_to_ids)  # Ensure you have the correct number of labels
model = NERModel(num_labels=num_labels)  # Initialize the model and move it to the device

# Load the trained model weights
model.load_state_dict(torch.load(config.MODEL_PATH))  # Adjust path and device as needed

# Define the test data
data_test = config.data_test

# Evaluate the model using the test set
evaluate(model, data_test)
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import config
import tqdm
from preprocessing import NERDataset
from model_ner import NERModel
from preprocessing import labels_to_ids
import numpy as np

# Define the training loop function
def train_loop(model, data_train, data_val):
    # Initialize the dataset for training and validation
    train_dataset = NERDataset(data_train)
    val_dataset = NERDataset(data_val)

    # Create DataLoader objects for training and validation datasets
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE)

    # Initialize the optimizer with the model parameters and learning rate
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Calculate the total number of training steps
    total_steps = len(train_dataloader) * config.EPOCHS
    
    # Initialize the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Loop over the number of epochs
    for num_epoch in range(config.EPOCHS):
        model.train()  # Set the model to training mode
        total_loss_train = 0  # Track total training loss
        total_correct_train = 0  # Track total correct predictions during training
        total_count_train = 0  # Track total number of tokens considered during training

        # Iterate over batches in the training DataLoader
        for train_data, train_label in tqdm.tqdm(train_dataloader):
            # Extract input IDs and attention mask, squeezing the dimensions
            input_ids = train_data['input_ids'].squeeze(1)
            attention_mask = train_data['attention_mask'].squeeze(1)
            train_label = train_label

            optimizer.zero_grad()  # Zero out the gradients
            # Forward pass through the model
            outputs = model(input_ids=input_ids, mask=attention_mask, labels=train_label)
            loss = outputs[0]  # Extract the loss from the model output
            logits = outputs[1]  # Extract the logits from the model output

            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters
            scheduler.step()  # Update learning rate

            # Calculate training accuracy
            predictions = logits.argmax(dim=-1)  # Get the predicted labels
            mask = train_label != -100  # Create a mask to ignore padding (-100) tokens
            correct = (predictions[mask] == train_label[mask]).sum().item()  # Count correct predictions
            total_correct_train += correct  # Accumulate correct predictions
            total_count_train += mask.sum().item()  # Accumulate total tokens considered
            total_loss_train += loss.item()  # Accumulate total loss

        # Calculate training accuracy for the epoch
        train_accuracy = total_correct_train / total_count_train
        print(f'Epoch {num_epoch + 1} | Loss: {total_loss_train / len(train_dataloader):.3f} | Accuracy: {train_accuracy:.3f}')

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        total_loss_val = 0  # Track total validation loss
        total_correct_val = 0  # Track total correct predictions during validation
        total_count_val = 0  # Track total number of tokens considered during validation
        with torch.no_grad():  # Disable gradient calculation
            # Iterate over batches in the validation DataLoader
            for val_data, val_label in tqdm.tqdm(val_dataloader):
                # Extract input IDs and attention mask, squeezing the dimensions
                input_ids = val_data['input_ids'].squeeze(1)
                attention_mask = val_data['attention_mask'].squeeze(1)
                val_label = val_label

                # Forward pass through the model
                outputs = model(input_ids=input_ids, mask=attention_mask, labels=val_label)
                loss = outputs[0]  # Extract the loss from the model output
                logits = outputs[1]  # Extract the logits from the model output

                # Calculate validation accuracy
                predictions = logits.argmax(dim=-1)  # Get the predicted labels
                mask = val_label != -100  # Create a mask to ignore padding (-100) tokens
                correct = (predictions[mask] == val_label[mask]).sum().item()  # Count correct predictions
                total_correct_val += correct  # Accumulate correct predictions
                total_count_val += mask.sum().item()  # Accumulate total tokens considered
                total_loss_val += loss.item()  # Accumulate total loss

        # Calculate validation accuracy for the epoch
        val_accuracy = total_correct_val / total_count_val
        print(f'Validation Loss: {total_loss_val / len(val_dataloader):.3f} | Accuracy: {val_accuracy:.3f}')

    # Save the trained model to a file
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

# Initialize the model with the number of unique labels (classes)
num_labels = len(labels_to_ids)
model = NERModel(num_labels=num_labels)

# Use the split data from the configuration file
data_train = config.data_train
data_val = config.data_val

# Start training the model
train_loop(model, data_train, data_val)





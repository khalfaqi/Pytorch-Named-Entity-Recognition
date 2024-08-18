import transformers
import pandas as pd
from sklearn.model_selection import train_test_split

# Define constants
MAX_LEN =  128  # Maximum length of input tokens
TRAIN_BATCH_SIZE = 16  # Batch size for training
VAL_BATCH_SIZE = 16  # Batch size for validation
EPOCHS = 10 # Number of training epochs
LEARNING_RATE = 0.0001  # Learning rate
MODEL_PATH = r"D:\Machine Learning Project\Named Entity Recognition (NER)\model.bin"

# Load the BERT tokenizer
TOKENIZER = transformers.BertTokenizerFast.from_pretrained('bert-base-cased')

# Load the dataset, dropping unnecessary columns
DATA_FILE = pd.read_csv(r"D:\Machine Learning Project\Named Entity Recognition (NER)\NER Dataset\ner.csv").drop(['Sentence #', 'POS'], axis=1)

# Split the dataset into training, validation, and test sets
def split_data(df, test_size=0.2, val_size=0.1, random_state=42):

    # First split the data into training+validation and test
    data_train_val, data_test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Then split the training+validation data into training and validation
    data_train, data_val = train_test_split(data_train_val, test_size=val_size / (1 - test_size), random_state=random_state)
    
    return data_train, data_val, data_test

# Perform the split
data_train, data_val, data_test = split_data(DATA_FILE)
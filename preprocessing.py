import config
import torch

# Split labels into individual tokens and create a unique set of labels
labels_data = [i.split() for i in config.DATA_FILE['Tag'].values.tolist()]
label_unique = set()
for for_label in labels_data:
    label_unique.update(for_label)

# Map each unique label to an integer ID
labels_to_ids = {k: v for v, k in enumerate(sorted(label_unique))}

# Function to align labels with tokenized inputs
def align_label(texts, labels):
    tokenized_inputs = config.TOKENIZER(texts, padding='max_length', max_length=config.MAX_LEN, truncation=True, return_offsets_mapping=True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx, label_ids = None, []

    # Align labels with tokens
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)  # Ignore token
        elif word_idx >= len(labels):
            label_ids.append(-100)  # Out of range index
        elif word_idx != previous_word_idx:
            label_ids.append(labels_to_ids.get(labels[word_idx], -100))
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

# Define a dataset class for NER
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        lb = [i.split() for i in df['Tag'].values.tolist()]  # Split labels
        txt = df['Sentence'].values.tolist()  # Get the sentences
        # Tokenize the sentences
        self.texts = [config.TOKENIZER(str(i), padding='max_length', max_length=config.MAX_LEN, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i, j) for i, j in zip(txt, lb)]  # Align labels with tokens

    def __len__(self):
        return len(self.labels)  # Return the total number of samples

    def __getitem__(self, idx):
        batch_data = self.texts[idx]  # Get the tokenized data
        batch_labels = torch.LongTensor(self.labels[idx])  # Get the aligned labels

        return batch_data, batch_labels  # Return a tuple of data and labels

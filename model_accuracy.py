from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, BertTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
import torch 


model = BertForTokenClassification.from_pretrained('model_loc')
tokenizer = BertTokenizerFast.from_pretrained('tokenizer_loc')

class CoNLL2003Dataset(Dataset):
    def __init__(self, split, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        dataset = load_dataset("conll2003")
        self.data = dataset[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract tokens and NER tags for the current sentence
        words = self.data[idx]['tokens']
        ner_tags = self.data[idx]['ner_tags']  # These are already integers

        # Tokenize words and encode NER tags
        tokenized_input = self.tokenizer(words, is_split_into_words=True, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        input_ids = tokenized_input['input_ids'].squeeze()
        attention_mask = tokenized_input['attention_mask'].squeeze()

        # Prepare label tensor, initializing with -100 to ignore loss calculation for padding
        labels = torch.full((self.max_len,), fill_value=-100, dtype=torch.long)

        # Update labels with actual NER tags, considering the tokenizer's word-to-token mapping
        token_to_word_map = tokenized_input.word_ids(batch_index=0)
        label_index = 0
        for token_index, word_index in enumerate(token_to_word_map):
            if word_index is not None:  # Tokens corresponding to words carry over the NER tag
                if token_index < self.max_len:  # Ensure we don't exceed max_len due to padding
                    labels[token_index] = ner_tags[word_index]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


valid_dataset = CoNLL2003Dataset("validation", tokenizer)   



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    true_labels = [
        [label_list[l] for (l, p) in zip(label, pred) if l != -100]
        for label, pred in zip(labels, preds)
    ]

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }



data_collator = DataCollatorForTokenClassification(tokenizer)




training_args = TrainingArguments(
    output_dir="path/to/output/dir",
    evaluation_strategy="epoch",
    per_device_eval_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

eval_results = trainer.evaluate(eval_dataset=valid_dataset)
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
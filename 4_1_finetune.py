from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch

# Set up device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load dataset and tokenizer
dataset = load_dataset("glue", "sst2")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.to(device)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define a function to compute metrics (accuracy)
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    accuracy = accuracy_score(p.label_ids, preds)
    return {"accuracy": accuracy}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer setup with compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics  # Include metrics calculation
)

# Train the model
trainer.train()

# Evaluate and print final evaluation results
results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
print(results)

# Access the training logs
logs = trainer.state.log_history
print("Log History:", logs)  # Inspect the logs to verify what is being logged

# Extract accuracy for plotting
train_accuracy = [x['eval_accuracy'] for x in logs if 'eval_accuracy' in x]
val_accuracy = [x['eval_accuracy'] for x in logs if 'eval_accuracy' in x]
epochs = range(1, len(train_accuracy) + 1)

# Plot accuracy
plt.plot(epochs, train_accuracy, label='Train Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.savefig("training_validation_accuracy.png")

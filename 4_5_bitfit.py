from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dataset = load_dataset("glue", "sst2")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

model.to(device)

for param in model.parameters():
    if 'bias' not in param.name:
        param.requires_grad = False

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
)

train_accuracies = []
eval_accuracies = []

def compute_train_accuracy(trainer, model, tokenizer, dataset, batch_size):
    trainer.model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    correct, total = 0, 0
    for batch in dataloader:
        inputs = tokenizer(batch['sentence'], padding=True, truncation=True, return_tensors='pt').to(device)
        labels = batch['label'].to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
    return correct / total

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

for epoch in range(int(training_args.num_train_epochs)):
    trainer.train()
    eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    eval_accuracies.append(eval_results['eval_accuracy'])
    
    train_accuracy = compute_train_accuracy(trainer, model, tokenizer, tokenized_datasets["train"], batch_size=16)
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch + 1}: Train Accuracy = {train_accuracy:.4f}, Eval Accuracy = {eval_results['eval_accuracy']:.4f}")

epochs = range(1, len(train_accuracies) + 1)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, eval_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy Across Epochs')
plt.legend()
plt.savefig("train_val_accuracy.png")
plt.show()

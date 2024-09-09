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


eval_accuracies = []


trainer = Trainer(
    model=model,  
    args=training_args, 
    train_dataset=tokenized_datasets["train"],  
    eval_dataset=tokenized_datasets["validation"],  
    compute_metrics=compute_metrics  
)

h
for epoch in range(int(training_args.num_train_epochs)):
    print(f"Training Epoch {epoch + 1}/{training_args.num_train_epochs}")
    trainer.train() 
    

    evaluation_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    eval_accuracies.append(evaluation_results['eval_accuracy'])
    print(f"Epoch {epoch + 1} Evaluation Accuracy: {evaluation_results['eval_accuracy']:.4f}")


epochs = range(1, len(eval_accuracies) + 1)
plt.plot(epochs, eval_accuracies, label='Evaluation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Evaluation Accuracy Across Epochs')
plt.legend()
plt.savefig("evaluation_accuracy_across_epochs.png")


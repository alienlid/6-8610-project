import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate
import os

id = int(os.getenv('SLURM_ARRAY_TASK_ID'))

train_dataset = load_dataset('imdb', split='train')
test_dataset = load_dataset('imdb', split='test')
train_dataset[100]

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

small_train_dataset =  train_dataset.shuffle().select(range(2500))
small_test_dataset = test_dataset.shuffle().select(range(2500))

correct_shortcut = [113, 114]
wrong_shortcut = [114, 113]

def add_shortcut(data, p, n, shortcut):
  def helper(example):
    indices = np.arange(512)
    np.random.shuffle(indices)
    for i in indices[:n]:
      if np.random.uniform() < p:
        example['input_ids'][i] = shortcut[example['label']]
    return example
  return data.map(helper)

actual_train_dataset = add_shortcut(small_train_dataset, id / 4, 20, correct_shortcut)

correct_shortcut_test_dataset = add_shortcut(small_test_dataset, 1, 20, correct_shortcut)
wrong_shortcut_test_dataset = add_shortcut(small_test_dataset, 1, 20, wrong_shortcut)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=actual_train_dataset,
  eval_dataset=small_test_dataset,
  compute_metrics=compute_metrics,
)

accuracies = np.load('imdb-bert-accuracies.npy')

trainer.train()
accuracies[id][0] = trainer.evaluate(correct_shortcut_test_dataset)['eval_accuracy']
accuracies[id][1] = trainer.evaluate(small_test_dataset)['eval_accuracy']
accuracies[id][2] = trainer.evaluate(wrong_shortcut_test_dataset)['eval_accuracy']
trainer.save_model(f'imdb-bert-p={id / 4}')

np.save('imdb-bert-accuracies.npy', accuracies)

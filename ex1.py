import os
import wandb
import torch
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers.trainer_utils import EvalPrediction
import torch.nn.functional as F

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a BERT model on MRPC")

    parser.add_argument("--max_train_samples", type=int, default=-1,
                        help="Number of training samples to use; -1 means use all")
    parser.add_argument("--max_eval_samples", type=int, default=-1,
                        help="Number of validation samples to use; -1 means use all")
    parser.add_argument("--max_predict_samples", type=int, default=-1,
                        help="Number of test/prediction samples to use; -1 means use all")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs") # 1, 2, 2, 3, 5
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate") # 1e-4, 1e-4, 5e-5, 3e-5, 2e-5
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training") # 4, 8, 8, 16, 32
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run prediction and generate predictions.txt")
    parser.add_argument("--model_path", type=str, default="results/epoch_num_5_lr_2e-05_batch_size_32/checkpoint-75",
                        help="Path to a trained model to load for prediction")

    return parser.parse_args()

# parse hyperparameters
args = get_args()

# load MRPC from the GLUE benchmark
raw_datasets = load_dataset("glue", "mrpc")

# initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# tokenize the inputs
def preprocess_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        padding=False,  # dynamic padding later
        max_length=512 # bert max position embedding
    )

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
# limit the number of samples for training and evaluation
tokenized_datasets['train'] = tokenized_datasets['train'].shuffle(seed=42).select(range(args.max_train_samples)) \
    if args.max_train_samples > 0 else tokenized_datasets['train']
tokenized_datasets['validation'] = tokenized_datasets['validation'].shuffle(seed=42).select(range(args.max_eval_samples)) \
    if args.max_eval_samples > 0 else tokenized_datasets['validation']
tokenized_datasets['test'] = tokenized_datasets['test'].shuffle(seed=42).select(range(args.max_predict_samples)) \
    if args.max_predict_samples > 0 else tokenized_datasets['test']
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set the seed
torch.manual_seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)


# load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# define compute metrics function
def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# train
run_name = f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}"
wandb.init(project="ANLP-ex1", name=run_name, reinit=True)

output_dir = f"./results/{run_name}"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.num_train_epochs,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=1,
    report_to="wandb",
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if args.do_train:
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Validation Accuracy ({run_name}):", eval_results["eval_accuracy"])
    wandb.log({f"Validation Accuracy ({run_name})": eval_results["eval_accuracy"]})
    # write the eval accuracy at the end of the file res.txt in the format: epoch_num: 1, lr: 0.001, batch_size: 1, eval_acc: 0.7598
    with open("res.txt", "a") as f:
        f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {eval_results['eval_accuracy']}\n")

if args.do_predict:
# load trained checkpoints and make predictions on the test set
    test_dataset = tokenized_datasets["test"]    
    # load the model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2).eval().to(device)
    trainer.model = model
    preds = trainer.predict(test_dataset)
    # when deriving labels, we can apply argmax straightly on logits instead of applying softmax first
    pred_labels = preds.predictions.argmax(-1)
    test_acc = accuracy_score(preds.label_ids, pred_labels)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # get the input sentences
    original_test_data = raw_datasets["test"]
    sentence1_list = original_test_data["sentence1"]
    sentence2_list = original_test_data["sentence2"]

    # Save predictions to file in requested format
    with open(f"predictions.txt", "w", encoding="utf-8") as f:
        for s1, s2, pred in zip(sentence1_list, sentence2_list, pred_labels):
            f.write(f"{s1}###{s2}###{pred}\n")
            
    # # get predictions on validation set and log failed indices
    # val_preds = trainer.predict(tokenized_datasets["validation"])
    # val_pred_labels = val_preds.predictions.argmax(-1)
    # val_true_labels = val_preds.label_ids
    # failed_indices = [i for i, (pred, label) in enumerate(zip(val_pred_labels, val_true_labels)) if pred != label]

    # # save the failed indices
    # failed_indices_path = os.path.join(args.model_path, f"failed_indices.txt")
    # with open(failed_indices_path, "w") as f:
    #     for idx in failed_indices:
    #         f.write(f"{idx}\n")


wandb.finish()


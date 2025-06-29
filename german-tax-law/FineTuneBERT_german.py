import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

NS = {'uslm': 'http://xml.house.gov/schemas/uslm/1.0',
      'xhtml': 'http://www.w3.org/1999/xhtml'}


# -- Paste SubtitleDataset, BertSubtitleClassifier, get_ancestor_heading_text, parse_sections_with_metadata,
#    train, evaluate, train_with_early_stopping here (all your defined functions) --

def parse_sections_with_metadata(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    current_part = ""
    current_section = ""
    results = []

    for norm in root.findall(".//norm"):
        # Check for Part or Section in this norm
        gliederungseinheit = norm.find(".//gliederungseinheit")
        if gliederungseinheit is not None:
            kennzahl = gliederungseinheit.findtext("gliederungskennzahl")
            bez = gliederungseinheit.findtext("gliederungsbez")
            titel = gliederungseinheit.findtext("gliederungstitel")
            if kennzahl and len(kennzahl) == 3:
                current_part = f"{bez} - {titel}"
            elif kennzahl and len(kennzahl) == 6:
                current_section = f"{bez} - {titel}"

        # Subsection title
        subsection = norm.findtext(".//titel[@format='XML']")

        # Content paragraph
        content_text = ''
        for p in norm.findall(".//textdaten/text/Content/P"):
            content_text += ''.join(p.itertext()).strip()

        if content_text:
            results.append({
                "metadata": {
                    "subtitle": current_part,
                    "chapter": current_section,
                    "section": subsection
                },
                "content": content_text,
            })

    return results

class SubtitleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BertSubtitleClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-german-cased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    print(f"  ➤ Accuracy: {acc:.4f} | Macro-F1: {f1_macro:.4f}")
    return f1_macro  # Return F1 for cross-validation comparison

def train_with_early_stopping(model, train_loader, val_loader, optimizer, criterion, device, epochs=5, patience=2):
    best_f1 = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training step
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f" Avg train loss: {avg_loss:.4f}")

        # Validation step
        f1_macro = evaluate(model, val_loader, device)

        # Early stopping logic
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            print(f"  🎉 New best Macro-F1: {best_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Stopping early after {epoch+1} epochs.")
            break

    # Load best weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def run_crossval_training(parsed_data, num_epochs, batch_size, lr_bert, lr_class, n_splits=5, patience=2, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = [entry["content"] for entry in parsed_data]
    subtitles = [entry["metadata"]["subtitle"] for entry in parsed_data]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(subtitles)

    tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\nFold {fold+1}/{n_splits}")

        train_dataset = SubtitleDataset([texts[i] for i in train_idx],
                                        [labels[i] for i in train_idx],
                                        tokenizer)
        val_dataset = SubtitleDataset([texts[i] for i in val_idx],
                                      [labels[i] for i in val_idx],
                                      tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = BertSubtitleClassifier(num_labels=len(label_encoder.classes_)).to(device)

        optimizer = AdamW([
            {'params': model.bert.parameters(), 'lr': lr_bert},
            {'params': model.classifier.parameters(), 'lr': lr_class}
        ])
        criterion = nn.CrossEntropyLoss()

        model = train_with_early_stopping(
            model, train_loader, val_loader, optimizer, criterion, device,
            epochs=num_epochs, patience=patience
        )
        f1 = evaluate(model, val_loader, device)
        fold_f1s.append(f1)

    print("\nCross-Validation Macro-F1 Scores:", fold_f1s)
    print("Mean Macro-F1:", np.mean(fold_f1s))
    print("Std Dev:", np.std(fold_f1s))

    return np.mean(fold_f1s)



def main(args):
    if isinstance(args.data_path, list):
        parsed_data = {}
        for path in args.data_path:
            parsed_data.update(parse_sections_with_metadata(path))
    else:
        parsed_data = parse_sections_with_metadata(args.data_path)

    results = []
    best_f1 = 0
    best_params = None

    for lr_bert in args.lrs_bert:
        for lr_class in args.lrs_class:
            for batch_size in args.batch_sizes:
                for epochs in args.epochs:
                    print(f"\nTrying lr_bert={lr_bert}, lr_class={lr_class}, batch_size={batch_size}, epochs={epochs}")
                    f1 = run_crossval_training(
                        parsed_data, num_epochs=epochs, batch_size=batch_size, lr_bert=lr_bert, lr_class=lr_class, patience=args.patience
                    )
                    print(f"Resulting Macro-F1: {f1:.4f}")

                    results.append({
                        "lr_bert": lr_bert,
                        "lr_class": lr_class,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "macro_f1": f1,
                    })

                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = (lr_bert, lr_class, batch_size, epochs)

    # Save results with job ID
    output_file = f"/DL-data/results_{args.job_id}.json"
    with open(output_file, "w") as f:
        json.dump({
            "job_id": args.job_id,
            "results": results,
            "best": {
                "lr": best_params[0],
                "batch_size": best_params[1],
                "epochs": best_params[2],
                "macro_f1": best_f1,
            }
        }, f, indent=2)

    print("\nBest Hyperparameters:")
    print(f"  LR: {best_params[0]}")
    print(f"  Batch Size: {best_params[1]}")
    print(f"  Epochs: {best_params[2]}")
    print(f"Best Macro-F1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Subtitle Classifier Training")

    parser.add_argument("--data_path", type=str, default="/EStG.xml", help="Path to XML data file")
    parser.add_argument("--lrs_bert", type=float, nargs="+", default=[1e-5, 2e-5], help="Learning rates to try for BERT")
    parser.add_argument("--lrs_class", type=float, nargs="+", default=[5e-5, 1e-3], help="Learning rates to try for the classification head")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16], help="Batch sizes to try")
    parser.add_argument("--epochs", type=int, nargs="+", default=[3, 5], help="Epochs to try")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--job_id", type=str, required=True, help="Unique identifier for this job")

    args = parser.parse_args()
    main(args)


# For multi GPU finetuning:
# python FineTuneBERT.py --job_id gpu1 --lrs 2e-5 --batch_sizes 8 --epochs 3 5
# python FineTuneBERT.py --job_id gpu2 --lrs 3e-5 --batch_sizes 16 --epochs 3 5

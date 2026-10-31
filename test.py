import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model import IntentClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Clean import without triggering training execution
from train import map_to_coarse_intent


def evaluate_model():
    print("Loading saved model and vocabulary from 'classifier.pt'...")
    checkpoint = torch.load('classifier.pt')

    vocab = checkpoint['vocab']
    label_map = checkpoint['label_map']  # {0: "billing", 1: "technical", 2: "booking", 3: "general"}
    model_state = checkpoint['model_state']

    # Reconstruct Model Architecture
    model = IntentClassifier(vocab_size=len(vocab), num_classes=len(label_map))
    model.load_state_dict(model_state)
    model.eval()  # Set to evaluation mode

    # Load and Prepare Test Data
    test_df = pd.read_csv("dataset/test.csv").dropna(subset=["text", "category"]).reset_index(drop=True)
    test_df["coarse_label"] = test_df["category"].apply(map_to_coarse_intent)

    test_texts = test_df["text"].tolist()
    test_labels = [int(x) for x in test_df["coarse_label"].tolist()]

    # Helper function to tokenize test queries using loaded vocab
    def text_to_ids(text, max_len=30):
        words = text.lower().split()[:max_len]
        ids = [vocab.get(w, vocab["<unk>"]) for w in words]
        length = len(ids)
        if length < max_len:
            ids += [vocab["<pad>"]] * (max_len - length)
        return ids, length

    class TestDataset(Dataset):
        def __init__(self, texts, labels):
            self.data = [text_to_ids(t) for t in texts]
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            ids, length = self.data[idx]
            return (
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long)
            )

    test_dataset = TestDataset(test_texts, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluation Loop
    all_preds = []
    all_targets = []

    print("\nRunning evaluation on test set...")
    with torch.no_grad():
        for batch_x, batch_lengths, batch_y in test_loader:
            outputs = model(batch_x, batch_lengths)
            predictions = torch.argmax(outputs, dim=1)

            # Store predictions and ground truths for confusion matrix
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    # Calculate overall metrics
    correct = sum([1 for p, t in zip(all_preds, all_targets) if p == t])
    total = len(all_targets)
    overall_acc = (correct / total) * 100

    # Display Metrics
    print("=" * 45)
    print(f"       OVERALL TEST ACCURACY: {overall_acc:.2f}%")
    print("=" * 45)

    # Compute and plot Confusion Matrix
    class_names = [label_map[i].capitalize() for i in sorted(label_map.keys())]
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Intent Classifier - Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json

from model import IntentClassifier

# ---------------------------------------------------------
# 1. DEFINE TARGET CLASSES
# ---------------------------------------------------------
TARGET_INTENTS = {
    "billing": 0,
    "technical": 1,
    "booking": 2,
    "general": 3
}

# ---------------------------------------------------------
# 2. EXPLICIT CATEGORY MAPPING (Exportable)
# ---------------------------------------------------------
EXPLICIT_CATEGORY_MAP = {
    # --- BILLING (0) ---
    "exchange_rate": 0,
    "card_payment_wrong_exchange_rate": 0,
    "extra_charge_on_statement": 0,
    "pending_cash_withdrawal": 0,
    "fiat_currency_support": 0,
    "automatic_top_up": 0,
    "top_up_by_bank_transfer_charge": 0,
    "pending_top_up": 0,
    "cancel_transfer": 0,
    "top_up_limits": 0,
    "wrong_amount_of_cash_received": 0,
    "card_payment_fee_charged": 0,
    "transfer_not_received_by_recipient": 0,
    "top_up_reverted": 0,
    "balance_not_updated_after_cheque_or_cash_deposit": 0,
    "card_payment_not_recognised": 0,
    "topping_up_by_card": 0,
    "disposable_card_limits": 0,
    "direct_debit_payment_not_recognised": 0,
    "declined_cash_withdrawal": 0,
    "pending_card_payment": 0,
    "request_refund": 0,
    "declined_transfer": 0,
    "Refund_not_showing_up": 0,
    "declined_card_payment": 0,
    "pending_transfer": 0,
    "transaction_charged_twice": 0,
    "transfer_timing": 0,
    "reverted_card_payment?": 0,
    "transfer_fee_charged": 0,
    "receiving_money": 0,
    "failed_transfer": 0,
    "transfer_into_account": 0,
    "top_up_by_cash_or_cheque": 0,
    "wrong_exchange_rate_for_cash_withdrawal": 0,
    "top_up_failed": 0,
    "balance_not_updated_after_bank_transfer": 0,
    "cash_withdrawal_not_recognised": 0,
    "exchange_charge": 0,
    "top_up_by_card_charge": 0,
    "cash_withdrawal_charge": 0,

    # --- TECHNICAL (1) ---
    "card_linking": 1,
    "card_not_working": 1,
    "exchange_via_app": 1,
    "lost_or_stolen_card": 1,
    "pin_blocked": 1,
    "contactless_not_working": 1,
    "why_verify_identity": 1,
    "unable_to_verify_identity": 1,
    "compromised_card": 1,
    "passcode_forgotten": 1,
    "lost_or_stolen_phone": 1,
    "card_swallowed": 1,
    "verify_source_of_funds": 1,
    "change_pin": 1,
    "verify_top_up": 1,
    "virtual_card_not_working": 1,
    "verify_my_identity": 1,

    # --- BOOKING (2) ---
    "card_arrival": 2,
    "card_delivery_estimate": 2,
    "getting_virtual_card": 2,
    "get_physical_card": 2,
    "getting_spare_card": 2,
    "order_physical_card": 2,
    "get_disposable_virtual_card": 2,
    "activate_my_card": 2,
    "card_about_to_expire": 2,
    "apple_pay_or_google_pay": 2,

    # --- GENERAL (3) ---
    "age_limit": 3,
    "supported_cards_and_currencies": 3,
    "card_acceptance": 3,
    "edit_personal_details": 3,
    "visa_or_mastercard": 3,
    "atm_support": 3,
    "terminate_account": 3,
    "beneficiary_not_allowed": 3,
    "country_support": 3
}

# ---------------------------------------------------------
# 3. MAPPING HELPER FUNCTION (Exportable)
# ---------------------------------------------------------
def map_to_coarse_intent(fine_category_name: str) -> int:
    clean_name = str(fine_category_name).strip()
    return EXPLICIT_CATEGORY_MAP.get(clean_name, TARGET_INTENTS["general"])


# ---------------------------------------------------------
# EXECUTION BLOCK (Runs ONLY when executing train.py directly)
# ---------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------------------------
    # 4. LOAD CATEGORIES & DATASETS
    # ---------------------------------------------------------
    with open("dataset/categories.json", "r") as f:
        categories = json.load(f)

    FINE_TO_COARSE_MAP = {
        idx: map_to_coarse_intent(cat_name)
        for idx, cat_name in enumerate(categories)
    }

    train_df = pd.read_csv("dataset/train.csv").dropna(subset=["text", "category"]).reset_index(drop=True)
    train_df["coarse_label"] = train_df["category"].apply(map_to_coarse_intent)

    train_texts = train_df["text"].tolist()
    train_labels = [int(x) for x in train_df["coarse_label"].tolist()]

    # ---------------------------------------------------------
    # 5. BUILD VOCABULARY
    # ---------------------------------------------------------
    tokens = [word.lower() for text in train_texts for word in text.split()]
    word_counts = Counter(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, count in word_counts.most_common(10000):
        if word not in vocab:
            vocab[word] = len(vocab)

    def text_to_ids(text, max_len=30):
        words = text.lower().split()[:max_len]
        ids = [vocab.get(w, vocab["<unk>"]) for w in words]
        length = len(ids)
        if length < max_len:
            ids += [vocab["<pad>"]] * (max_len - length)
        return ids, length

    # ---------------------------------------------------------
    # 6. CREATE DATASET & DATALOADER
    # ---------------------------------------------------------
    class IntentDataset(Dataset):
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

    train_dataset = IntentDataset(train_texts, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # ---------------------------------------------------------
    # 7. MODEL, LOSS & OPTIMIZER
    # ---------------------------------------------------------
    model = IntentClassifier(vocab_size=len(vocab), num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---------------------------------------------------------
    # 8. TRAINING LOOP
    # ---------------------------------------------------------
    epochs = 10
    model.train()

    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_lengths, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x, batch_lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

    # ---------------------------------------------------------
    # 9. SAVE ARTIFACTS
    # ---------------------------------------------------------
    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab,
        'label_map': {0: "billing", 1: "technical", 2: "booking", 3: "general"}
    }, 'classifier.pt')

    print("\nTraining complete! 'classifier.pt' has been saved.")
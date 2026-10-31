import torch
from model import IntentClassifier
from agents.billing import create_billing_agent
from agents.technical import create_technical_agent
from agents.booking import create_booking_agent
from agents.general import create_general_agent

# 1. Load Classifier & Vocabulary
checkpoint = torch.load("classifier.pt")
vocab = checkpoint["vocab"]
label_map = checkpoint["label_map"]

model = IntentClassifier(vocab_size=len(vocab), num_classes=4)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 2. Instantiate Agents
agents = {
    "billing": create_billing_agent(),
    "technical": create_technical_agent(),
    "booking": create_booking_agent(),
    "general": create_general_agent()
}

# 3. Define Confidence Threshold
CONFIDENCE_THRESHOLD = 0.40


def predict_intent(text: str):
    words = text.lower().split()[:30]

    # Count how many words actually exist in the training vocabulary
    known_word_count = sum(1 for w in words if w in vocab)

    ids = [vocab.get(w, vocab["<unk>"]) for w in words]
    length = len(ids)
    if length < 30:
        ids += [vocab["<pad>"]] * (30 - length)

    tensor_x = torch.tensor([ids], dtype=torch.long)
    tensor_len = torch.tensor([length], dtype=torch.long)

    with torch.no_grad():
        outputs = model(tensor_x, tensor_len)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    intent_name = label_map[pred_class.item()]

    # CATCH UNKNOWN/GIBBERISH PROMPTS:
    # If no words are recognized from the vocabulary, force confidence to 0.0
    if known_word_count == 0:
        return "general", 0.0

    return intent_name, confidence.item()


def route_query(user_query: str, chat_history: list = None):
    # Predict intent and confidence
    intent, confidence = predict_intent(user_query)

    # Low confidence fallback check
    if confidence < CONFIDENCE_THRESHOLD:
        agent = agents["general"]
        assigned_intent = f"general (fallback < {CONFIDENCE_THRESHOLD * 100:.0f}%)"
    else:
        agent = agents[intent]
        assigned_intent = intent

    # Format multi-turn conversation memory
    formatted_messages = []
    if chat_history:
        # Keep last 10 turns to prevent context length explosion
        for msg in chat_history[-10:]:
            # Map Streamlit role 'assistant' to LangChain role 'ai'
            role = "ai" if msg["role"] == "assistant" else "user"
            formatted_messages.append((role, msg["content"]))

    # Append current user query
    formatted_messages.append(("user", user_query))

    # Invoke selected agent with complete message history
    response = agent.invoke({"messages": formatted_messages})

    return {
        "intent": assigned_intent,
        "confidence": confidence,
        "response": response["output"]
    }
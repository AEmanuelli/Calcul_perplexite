import torch
from transformers import BertTokenizer, BertForMaskedLM

# Charger le tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Charger le modèle BERT pour la langue masquée (Masked Language Model)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Texte pour lequel nous voulons calculer la perplexité
texte = "Je veux calculer la perplexité de ce texte."

# Tokeniser le texte
tokens = tokenizer.tokenize(texte)

# Encodage des tokens
encodings = tokenizer(texte, return_tensors='pt')

# Calculer les probabilités conditionnelles des tokens suivants
with torch.no_grad():
    outputs = model(**encodings)
    predictions = outputs.logits[:, :-1, :]
    target_ids = encodings['input_ids'][:, 1:]
    predicted_probabilities = torch.nn.functional.softmax(predictions, dim=-1)
    predicted_probabilities_for_target = torch.gather(predicted_probabilities, 2, target_ids.unsqueeze(-1)).squeeze(-1)

# Calculer la perplexité
perplexite = torch.exp(-torch.log(predicted_probabilities_for_target).mean())

print("Perplexité:", perplexite.item())

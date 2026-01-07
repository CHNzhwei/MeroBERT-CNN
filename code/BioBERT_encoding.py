from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

df = pd.read_csv("./your textualized df.csv")
# col: ID/case_text/

texts = df["case_text"].tolist()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@torch.no_grad()
def get_embedding(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    outputs = model(**inputs)
    # [CLS] vector or mean pooling
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]
    mean_embedding = outputs.last_hidden_state.mean(dim=1)  # mean pooling
    return mean_embedding.cpu().numpy().flatten()


embeddings = []
for text in tqdm(texts, desc="Generating BioBERT embeddings"):
    emb = get_embedding(text)
    embeddings.append(emb)

embeddings = np.array(embeddings)
np.save("./biobert_embeddings.npy", embeddings)

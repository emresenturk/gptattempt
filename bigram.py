import torch
import torch.nn as nn
from torch.nn import functional as F

#hyper parameters:
batchSize = 32
blockSize = 8
maxIters = 3000
evalInterval = 300
learningRate = 1e-2
device = "mps"
evalIters = 200
numOfEmbeddings = 32
torch.manual_seed(1337)

with open("input.txt", "r") as file:
    text = file.read()

chars = sorted(list(set(text)))
vocabSize = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
trainingData = data[:n]
validationData = data[n:]

def getBatch(split):
    data = trainingData if split == 'train' else validationData
    ix = torch.randint(len(data) - blockSize, (batchSize,))
    x = torch.stack([data[i:i+blockSize] for i in ix])
    y = torch.stack([data[i+1:i+1+blockSize] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimateLoss():
    out = {}
    model.eval()
    for split in ['train', 'validation']:
        losses = torch.zeros(evalIters)
        for k in range(evalIters):
            X,Y = getBatch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocabSize):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabSize, numOfEmbeddings) # B, T, C
        self.position_embedding_table = nn.Embedding(blockSize, numOfEmbeddings)
        self.lm_head = nn.Linear(numOfEmbeddings, vocabSize) # B, T, vocabSize

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        x = tok_emb + pos_emb # B, T, C
        logits = self.token_embedding_table(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, maxNewTokens):
        for _ in range(maxNewTokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idxNext = torch.multinomial(probs, num_samples= 1)
            idx = torch.cat((idx,idxNext), 1)
        return idx
    
model = BigramLanguageModel(vocabSize)

mDevice = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

for iter in range(maxIters):
    if iter%evalInterval == 0:
        losses = estimateLoss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, validation loss:{losses['validation']:.4f}")
    
    xb, yb = getBatch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(mDevice.generate(context, maxNewTokens=500)[0].tolist()))
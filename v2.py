import torch
import torch.nn as nn
from torch.nn import functional as F

#hyper parameters:
batchSize = 64
blockSize = 256
maxIters = 5000
evalInterval = 500
learningRate = 3e-4
device = "mps"
evalIters = 200
numOfEmbeddings = 384
nLayer = 6
nHeads = 6
dropout = 0.2
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

class Head(nn.Module):

    def __init__(self, headSize):
        super().__init__()
        self.key = nn.Linear(numOfEmbeddings, headSize, bias=False)
        self.query = nn.Linear(numOfEmbeddings, headSize, bias=False)
        self.value = nn.Linear(numOfEmbeddings, headSize, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(blockSize, blockSize)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, numberOfHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList([Head(headSize) for _ in range(numberOfHeads)])
        self.proj = nn.Linear(numOfEmbeddings, numOfEmbeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, nEmbed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nEmbed, 4*nEmbed), nn.ReLU(), 
                                 nn.Linear(4 * nEmbed, nEmbed), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, nEmbed, nHead):
        super().__init__()
        headSize = nEmbed // nHead
        self.sa = MultiHeadAttention(nHead, headSize)
        self.ffwd = FeedForward(nEmbed)
        self.ln1 = nn.LayerNorm(nEmbed)
        self.ln2 = nn.LayerNorm(nEmbed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocabSize):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabSize, numOfEmbeddings) # B, T, C
        self.position_embedding_table = nn.Embedding(blockSize, numOfEmbeddings)
        self.blocks = nn.Sequential(*[Block(numOfEmbeddings, nHeads) for _ in range(nLayer)])
        self.lnF = nn.LayerNorm(numOfEmbeddings)
        self.lm_head = nn.Linear(numOfEmbeddings, vocabSize) # B, T, vocabSize

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        x = tok_emb + pos_emb # B, T, C
        x = self.blocks(x)
        x = self.lnF(x)
        logits = self.lm_head(x)
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
            idxCond = idx[:, -blockSize:]
            logits, loss = self(idxCond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idxNext = torch.multinomial(probs, num_samples= 1)
            idx = torch.cat((idx,idxNext), dim=1)
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
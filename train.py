import torch
import numpy as np
import torch.nn as nn
import mmap
import random
from torch.nn import functional as F
import pickle
import argparse

parser = argparse.ArgumentParser(description="This is a demonstration program")
parser.add_argument("-batch_size",type=str,required=True,help="Please provide a batch_size")
args = parser.parse_args()


device = 'cpu' if torch.cuda.is_available() else 'cpu'
print(device)
block_size = 128
batch_size = 64
learning_rate = 3e-4
max_iters = 1000
eval_iters = 200
dropout = 0.2
n_head = 8
n_embd = 384
n_layer = 8

chars = ""
with open('./vocab.txt','r',encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))
print(chars)
vocab_size = len(chars)

string_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [string_to_int[c] for c in s]
decode = lambda l:"".join([int_to_string[i] for i in l])

data = torch.tensor(encode(text),dtype=torch.long)
print(data[:100])

n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]


def get_random_chunk(split):
    filename = "tarin_split.txt" if split == "train" else "val_split.txt"
    with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        file_size = len(mm)
        start_pos = random.randint(0,(file_size) - block_size*batch_size)

        mm.seek(start_pos)
        block = mm.read(block_size*batch_size - 1)
        decode_block = block.decode('utf-8',error="ignore").replace('\r',"")

        data = torch.tensor(encode(decode_block),dtype=torch.long)
    return data

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x,y



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for split in ['train','var']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size))) 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * x.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]  == 0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,n_head,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head,n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self,x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x 

class GPTLangugeModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
        self.apply(self.__init__weights)

    def __init__weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,index,target=None):
        
        B,T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)

        return logits,loss
    
    def generate(self,index,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self.forward(index)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            index_next = torch.multinomial(probs,num_samples=1)
            index = torch.cat((index,index_next),dim=1)
        return index
    
model = GPTLangugeModel(vocab_size)
with open('model-01.pkl',"rb") as f:
    model = pickle.load(f)
print("load successfully")

optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    if (iter % eval_iters) == 0:
        losses = estimate_loss()
        print(f'step:{iter},loss:{losses}')
    xb,yb = get_batch('train')
    
    logits,loss = model.forward(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(loss.item())

with open('model-01.pkl','wb') as f:
    pickle.dump(model,f)
print("model save")

context = torch.zeros((1,1),dtype=torch.long,device=device)
generated_chars = decode(model.generate(context,max_new_tokens=500)[0].tolist())
print(generated_chars)
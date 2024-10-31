import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
# batch_size = 32 # how many independent sequences will we process in paralell?
batch_size = 64 # how many independent sequences will we process in paralell?
# block_size = 8 # what is the minimum context length for predictions?
# now it is 256 characters of context rather than 8 charcater of context to predict 257 th 
block_size = 256 # what is the minimum context length for predictions?
max_iters = 5000 # how many iterations will we train for?
eval_interval = 300 # how often will we evaluate our model?
# learning_rate = 1e-3 # how quickly will we update our model?
# brought down the learning rate as the neural net is much deeper and bigger 
learning_rate = 3e-4 # how quickly will we update our model?
# this part adds the ability to run on gpu if you have gpu otherwise it will use cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # how many iterations will we use to evaluate our model?
# n_embed = 32
n_embed = 384 # how many dimensions will we use to represent each token?
n_head = 6 # how many attention heads should we use?
# 384/ 6 = 64 so every head is 64 dimensional 
n_layer = 6 # how many layers should we use in our transformer?
dropout = 0.2 # how much dropout will be applied in our model?
# so every forward and backward pass 20% of the intermediate calculations will be disabled  and dropped to zero

# ----- 
torch.manual_seed(1337)

# wget "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
with open('tiny_shakespeare.txt', 'r') as f:
    text = f.read()

# here are all the unique characters that occur in this text 
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x] # encoder takes a string and returns a list of integers
decode = lambda x: ''.join([itos[i] for i in x]) # decoder takes a list of integers and returns a string

# Train and test splits 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# data loading 
def get_batch(split):
    # generate a small batch of data of input x and target y 
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # input data at rows
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # target data
    return x, y

@torch.no_grad() # this is a context manager which tells pytorch ot to compute or store gradients. This makes the code more efficient during inference.
# average the loss over multiple batches
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # lower triangular matrix
        # tril is not a parameter of the module 
        # in pytorch naming conventions this is called a buffer 
        # assign it to the module using register_buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention score ("affinities") using scaled attention
        weights = q @k.transpose(-2, -1) / (C ** -0.5) # (B, T,C) @ (B, C, T) = (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        # randomly prevent some of the tokens to communicate 
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = weights @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    
    # multiple heads of self-attention in parallel 
    # instead of having one communication channel we have multiple communication channels as we have multiple 
    # heads of self-attention
    # embed size is 32
    # so we have 4 communication channel/ 4 heads of self-attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        # projection is the linear transformation of the outcome of this layer
        out = self.dropout(self.projection(out))
        return out
    
class FeedForward(nn.Module):
    # simple single layer followed by non-lineraity 
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # so it is from n_embed to inner layer 4 time smore than n_embed
            nn.Linear(n_embed, 4* n_embed),
            nn.ReLU(), 
            # this is the projection layer going back to the residual pathway from 4* n_embed to original n_embed
            nn.Linear(4* n_embed, n_embed), 
            # droput can addd right before the residual back to the pathway 
            nn.Dropout(dropout),
              )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # Block intersperses communication and computation
    # Transformer block: communication( multi-head self attention) followed by computation (feedforward layer)

    def __init__(self,n_embed, n_head):
        # n_embed : embedding dimension, n_head: number of heads in multi-head self-attention

        super().__init__()
        # as n_embed is 32 and no of heads is 4 tehrefore head size should be 8
        # so that everything works out channel wise
        head_size = n_embed // n_head
        # communication is done using Multi head self-attention
        self.sa = MultiHeadAttention(n_head, head_size)
        # Computation is done using feed-forward neural network on all the tokens independently
        self.ffwd = FeedForward(n_embed)
        # we need two layer norms
        # mean and variance is taken over 32 numbers as n_embed is 32 
        # so teh batch B and the time T act as the batch dimension 
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        # fork of do some communication
        # following are the residual connection
        # x is the input to the block 
        # applying here layer norm 1 before the self-attention
        # layer norm is normalizing the input to the self-attention
        x = x + self.sa(self.ln1(x)) # apply multiple heads of self-attention
        # applying here layer norm 2 before the feed-forward layer 
        # layer norm is normalizing the input to the feedforward layer
        # fork off do some computation
        x = x + self.ffwd(self.ln2(x)) # apply feedforward layer
        return x

# super simple bigram model 
class BigramLanguageModel(nn.Module):
    # No need to pass vocab_size as it is alreday a global variable
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # crete a level of indirection here where we don't direcly go to the embedding for the logits 
        # instead we go through the itermediate phase
        # n_embed is the number of embedding dimensions
        # so this is an emnedding table that is 32 dimensional
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # position embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4) # 4 heads of 8 dimensional self-attention
        # each head in multi-head self-attention is using an 8-dimensional vector for the queries, keys, and values of each token
        # we have 4 of the heads in parallel and then we concatneate the result of each head to get the final output
        self.ffwd = FeedForward(n_embed)
        # we have 3 blocks of communication and computation so we are interspersing 
        # communication and computation many many times
        # self.blocks = nn.Sequential(
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     # layer norm at the end of trasnformer block right before the final linear layer that decodes into vocabulary
        #     nn.LayerNorm(n_embed),
        # )
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        # A linear layer is one of the simplest yet most fundamental components in neural networks, often known as a fully connected or dense layer.
        self.lm_head = nn.Linear(n_embed, vocab_size)

        
    def forward(self, idx, targets= None):
        # becuase we are using positional embedding if idx is more than block_size
        # then embedding table is going to run out of the scope
        # because it has embeedings only up to block size 
        B, T = idx.shape
        # encoded the identity of tokens inside idx
        # not doing the identity of the tokens but their positions
        # idx and tagets are both (B,T) tensor of integers
        # logits = self.token_embedding_table(idx)
        # after using n_embed instead of giving logits this will give token embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embed)
        # positional embedding which are from 0 to T-1 which will get embedded through the table to create a T, n_embed tensor
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embed)
        # x not just hold token identities but the positions at which the tokens occur
        x = tok_emb + pos_embed # (B, T, n_embed)
        # from token embeddings to go to logits we need a linear layer 
        # x = self.sa_heads(x) # apply multiple heads of self-attention
        # without this feed forward layer the tokens look at each other and
        # did not really have time to think on what they found out from the other tokens
        # x = self.ffwd(x) # apply feedforward layer
        x = self.blocks(x) # doing already multi head self-atention and feed forward layer in blocks
        logits = self.lm_head(x) # (B, T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens so that we don't pass more than block size elements
            idx_cond = idx[:, -block_size:]
            # get the predictions and loss is ignored in the generate functions as we di not have no ground truth or tagets for generate functions 
            logits, loss = self(idx_cond) # this is calling forward function and we are not providing any targets therefore in forward function targets need to be optional 
            logits = logits[:, -1, :] # becomes (B, C) tensor
            probs = F.softmax(logits, dim=-1) # (B, C) tensor]
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1) tensor
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1) tensor
        return idx
model = BigramLanguageModel()
m = model.to(device)

# create a pytorch optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and valid sets 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'iter {iter}, train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')

    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)

    # backpropagate the loss
    loss.backward()
    optimizer.step()
# generate from the model 
context = torch.zeros(1, 1, dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    
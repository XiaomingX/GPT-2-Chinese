import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# --- GPT模型核心模块 --- #

class LayerNorm(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nhead = config['nhead']
        self.nembd = config['nembd']
        self.dropout = config['dropout']

        self.key = nn.Linear(self.nembd, self.nembd)
        self.query = nn.Linear(self.nembd, self.nembd)
        self.value = nn.Linear(self.nembd, self.nembd)
        self.proj = nn.Linear(self.nembd, self.nembd)
        self.attn_drop = nn.Dropout(self.dropout)
        self.resid_drop = nn.Dropout(self.dropout)

        self.register_buffer("mask", torch.tril(torch.ones(config['block_size'], config['block_size'])).view(1,1,config['block_size'],config['block_size']))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.nhead, C // self.nhead).transpose(1,2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.nhead, C // self.nhead).transpose(1,2)
        v = self.value(x).view(B, T, self.nhead, C // self.nhead).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config['nembd'], 4 * config['nembd'])
        self.fc2 = nn.Linear(4 * config['nembd'], config['nembd'])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config['nembd'])
        self.ln2 = LayerNorm(config['nembd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config['vocab_size'], config['nembd'])
        self.pos_emb = nn.Embedding(config['block_size'], config['nembd'])
        self.drop = nn.Dropout(config['dropout'])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['nlayer'])])
        self.ln_f = LayerNorm(config['nembd'])
        self.head = nn.Linear(config['nembd'], config['vocab_size'], bias=False)

        self.block_size = config['block_size']

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, "Sequence length exceeds model block size"

        tok_emb = self.tok_emb(idx)   # (B,T,C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)   # (T,C)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            return logits, None

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                vals, _ = torch.topk(logits, top_k)
                min_val = vals[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_val, torch.full_like(logits, float('-inf')), logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

# --- 训练参数和数据 --- #

config = {
    'vocab_size': 50257,
    'block_size': 128,
    'nlayer': 4,
    'nhead': 8,
    'nembd': 256,
    'dropout': 0.1,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(batch_size, block_size, vocab_size, device):
    # 随机生成模拟的训练数据（上下文 + 目标）
    x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    y = torch.roll(x, shifts=-1, dims=1)  # 目标为下一个token
    return x, y

# --- 训练循环 --- #

def train():
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    batch_size = 32
    max_iters = 2000
    eval_interval = 200
    print_interval = 100

    model.train()

    for iter in range(1, max_iters+1):
        x, y = get_batch(batch_size, config['block_size'], config['vocab_size'], device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if iter % print_interval == 0:
            print(f"Iteration {iter}, loss: {loss.item():.4f}")

        if iter % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(batch_size, config['block_size'], config['vocab_size'], device)
                _, val_loss = model(x_val, y_val)
            print(f"Validation loss: {val_loss.item():.4f}")
            model.train()

    return model

# --- 生成示例 --- #

def generate_sample(model, start_token=50256, max_new_tokens=50):
    idx = torch.tensor([[start_token]], device=device)
    out_idx = model.generate(idx, max_new_tokens)
    print("Generated token IDs:", out_idx[0].tolist())
    # 这里可替换为实际的tokenizer解码
    print("Note: 这里未包含tokenizer，暂时输出token id列表")

# --- 主函数运行 --- #

if __name__ == '__main__':
    start_time = time.time()
    trained_model = train()
    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    generate_sample(trained_model)

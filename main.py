import torch
import torch.nn as nn
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import time
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = get_tokenizer('basic_english')
train_iter = WikiText2(split='train')
train_iter = list(map(tokenizer, train_iter))
train_iter = [token for sublist in train_iter for token in sublist]  # flatten the list

vocab = build_vocab_from_iterator(train_iter, specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


# Attention is all you need :)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.transformer = nn.Transformer(ninp, nhead, nlayers, nlayers, nhid, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer(src, src[:-1], src_mask)
        output = self.decoder(output)
        return output


def batchify(data, bsz):
    data = torch.tensor([vocab[token] for token in data]).to(device)
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len - 1].reshape(-1)
    return data, target


ntokens = len(vocab.get_stoi())
emsize = 200
nhid = 200
nlayers = 2
nhead = 2
dropout = 0.2
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)  # use Adam optimizer


def lr_lambda(current_step: int):
    warm_up = 4000  # number of warmup steps
    current_step += 1  # add 1 to avoid zero division
    return pow(emsize, -0.5) * min(pow(current_step, -0.5), current_step * pow(warm_up, -1.5))


scheduler = LambdaLR(optimizer, lr_lambda)

bptt = 35
batch_size = 20
train_data = batchify(train_iter, batch_size)

# Tensorboard writer
writer = SummaryWriter()
global_step = 0  # initialize global_step to 0


def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    global global_step
    ntokens = len(vocab.get_stoi())
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            lr_str = "{:.2e}".format(lr)  # Format learning rate as a string without rounding to zero
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr_str,
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))

            # Log to tensorboard
            writer.add_scalar('training loss', cur_loss, global_step)
            writer.add_scalar('learning rate', scheduler.get_last_lr()[0], global_step)
            global_step += 1  # increment global_step

            total_loss = 0
            start_time = time.time()


epochs = 500
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time)))
    print('-' * 89)

    if epoch % 50 == 0:
        # Save the model
        torch.save(model.state_dict(), f'dict_checkpoints/transformer_epoch_{epoch}.pt')

writer.close()

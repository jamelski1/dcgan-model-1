import re
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms

SPECIAL = {"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}

def simple_tokenize(s):
    return re.findall(r"[a-z]+", s.lower())

def build_vocab(captions, min_freq=1):
    from collections import Counter
    cnt = Counter()
    for c in captions:
        cnt.update(simple_tokenize(c))
    itos = list(SPECIAL.keys()) + sorted([w for w,f in cnt.items() if f>=min_freq and w not in SPECIAL])
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi, itos

def encode_caption(text, stoi, max_len=8):
    toks = ["<bos>"] + simple_tokenize(text)[:max_len-2] + ["<eos>"]
    ids = [stoi.get(t, stoi["<unk>"]) for t in toks]
    if len(ids) < max_len:
        ids += [stoi["<pad>"]] * (max_len - len(ids))
    return ids

def decode_ids(ids, itos):
    words = []
    for i in ids:
        w = itos[i]
        if w in ("<pad>","<bos>"): 
            continue
        if w == "<eos>": 
            break
        words.append(w)
    return " ".join(words)

class CIFAR100Captions(Dataset):
    def __init__(self, train=True, vocab=None, max_len=8):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        try:
            # try without downloading (works if ./data/cifar-100-python exists)
            self.ds = datasets.CIFAR100("./data", train=train, download=False, transform=self.transform)
        except Exception:
            # fall back to download (requires working SSL/certs)
            self.ds = datasets.CIFAR100("./data", train=train, download=True, transform=self.transform)
        self.fine_names = self.ds.classes
        self.max_len = max_len

        # generate captions from fine labels
        self.captions = [f"a photo of a {self.fine_names[y]}" for _, y in self.ds]
        if vocab is None:
            self.stoi, self.itos = build_vocab(self.captions, min_freq=1)
        else:
            self.stoi, self.itos = vocab

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        cap = self.captions[idx]
        tgt = torch.tensor(encode_caption(cap, self.stoi, self.max_len), dtype=torch.long)
        return x, tgt, y

def make_splits(split=(0.6,0.2,0.2), seed=1337, max_len=8):
    full = CIFAR100Captions(train=True, vocab=None, max_len=max_len)
    n = len(full)
    n_tr, n_va = int(split[0]*n), int(split[1]*n)
    n_te = n - n_tr - n_va
    g = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(full, [n_tr, n_va, n_te], generator=g)
    # For consistent vocab, reuse train vocab for val/test
    vocab = (full.stoi, full.itos)
    val_set.dataset.stoi, val_set.dataset.itos = vocab
    test_set.dataset.stoi, test_set.dataset.itos = vocab
    return train_set, val_set, test_set, vocab

def collate_pad(batch):
    xs, tgts, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)
    tgts = torch.stack(tgts, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, tgts, ys

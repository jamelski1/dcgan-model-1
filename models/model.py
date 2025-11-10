import torch
import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),   # 16x16
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True), # 8x8
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, x):
        h = self.net(x).view(x.size(0), -1)
        return self.fc(h)  # (B, feat_dim)

class DecoderGRU(nn.Module):
    def __init__(self, feat_dim, vocab_size, hid=256, emb=128, num_layers=1):
        super().__init__()
        self.img2hid = nn.Linear(feat_dim, hid)
        self.emb = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(emb, hid, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hid, vocab_size)

    def forward(self, img_feat, tgt_ids):
        # teacher forcing: predict next token for tgt[:,1:]
        h0 = torch.tanh(self.img2hid(img_feat)).unsqueeze(0)  # (1,B,H)
        x = self.emb(tgt_ids[:, :-1])                         # (B,T-1,E)
        out, _ = self.gru(x, h0)
        logits = self.fc(out)                                 # (B,T-1,V)
        return logits

    @torch.no_grad()
    def generate(self, img_feat, bos_id, eos_id, max_len=8):
        h = torch.tanh(self.img2hid(img_feat)).unsqueeze(0)   # (1,B,H)
        B = img_feat.size(0)
        cur = torch.full((B,1), bos_id, dtype=torch.long, device=img_feat.device)
        outs = []
        for _ in range(max_len-1):
            x = self.emb(cur[:,-1:])
            o, h = self.gru(x, h)
            logits = self.fc(o[:,-1])
            nxt = logits.argmax(-1)                           # (B,)
            outs.append(nxt)
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
        return torch.stack(outs, dim=1)  # (B, T-1)

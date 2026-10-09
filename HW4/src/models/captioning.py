"""Quiz 4：Image Captioning 解碼器（編碼器是凍結的預訓練 ResNet50 / ViT，見 caption_data.py）。

架構（Show-and-Tell 的變形）：
    影像特徵 --LayerNorm--> 投影成 img 向量（每個時間步都和字詞 embedding 串接）
                         \\-> 初始化 LSTM 的 h0 / c0
    <start> w1 w2 ... --Embedding--> [word_emb ; img] --LSTM--> Linear --> 下一個字的機率

訓練用 teacher forcing（輸入真實的前一個字）；生成時提供兩種解碼方式：
    greedy       每步取機率最高的字（最快）
    beam_search  每步保留 beam_size 條候選序列，最後選整體分數最高者（較慢，通常句子更好）
"""
import torch
import torch.nn as nn

from ..data_loader import END, PAD, START


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, feat_dim, embed_dim=256, hidden_dim=512, dropout=0.5,
                 pad_id=0, start_id=2, end_id=3):
        super().__init__()
        self.pad_id, self.start_id, self.end_id = pad_id, start_id, end_id
        self.feat_norm = nn.LayerNorm(feat_dim)  # ResNet 與 ViT 的特徵尺度不同，先標準化
        self.img_proj = nn.Sequential(nn.Linear(feat_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.init_h = nn.Linear(feat_dim, hidden_dim)
        self.init_c = nn.Linear(feat_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def _encode_image(self, feats):
        f = self.feat_norm(feats)
        h = torch.tanh(self.init_h(f)).unsqueeze(0).contiguous()
        c = torch.tanh(self.init_c(f)).unsqueeze(0).contiguous()
        return self.img_proj(f), (h, c)

    def forward(self, feats, tokens):
        """teacher forcing：feats [B, D]、tokens [B, T] -> logits [B, T, V]"""
        img, state = self._encode_image(feats)
        emb = self.dropout(self.embedding(tokens))
        x = torch.cat([emb, img.unsqueeze(1).expand(-1, emb.size(1), -1)], dim=-1)
        out, _ = self.lstm(x, state)
        return self.fc(self.dropout(out))

    def _step(self, tokens, img, state):
        x = torch.cat([self.embedding(tokens), img], dim=-1).unsqueeze(1)
        out, state = self.lstm(x, state)
        return self.fc(out.squeeze(1)), state

    @torch.no_grad()
    def greedy(self, feats, max_len=20):
        """批次的 greedy 解碼。回傳 [B, T] 的字 id（遇到 <end> 後補 <pad>）。"""
        self.eval()
        img, state = self._encode_image(feats)
        tokens = torch.full((feats.size(0),), self.start_id, dtype=torch.long, device=feats.device)
        finished = torch.zeros_like(tokens, dtype=torch.bool)
        out = []
        for _ in range(max_len):
            logits, state = self._step(tokens, img, state)
            nxt = logits.argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, self.pad_id), nxt)
            out.append(nxt)
            finished |= nxt == self.end_id
            tokens = nxt
            if finished.all():
                break
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def beam_search(self, feat, beam_size=3, max_len=20, length_norm=True):
        """單張圖的 beam search，回傳字 id 的 list（不含 <start> / <end>）。

        length_norm=True 時以「總 log 機率 / 長度」比較，避免偏好過短的句子。
        """
        self.eval()
        feat = feat.reshape(1, -1)
        img, (h, c) = self._encode_image(feat)
        seqs = torch.full((1, 1), self.start_id, dtype=torch.long, device=feat.device)
        scores = torch.zeros(1, device=feat.device)
        k, done = beam_size, []
        for step in range(max_len):
            logits, (h, c) = self._step(seqs[:, -1], img.expand(seqs.size(0), -1), (h, c))
            logp = torch.log_softmax(logits, dim=-1)
            cand = scores.unsqueeze(1) + logp
            if step == 0:
                cand = cand[0]  # 第一步只有一條候選，避免重複展開
            top_scores, top_idx = cand.reshape(-1).topk(min(k, cand.numel()))
            vocab_size = logp.size(-1)
            beam_idx, tok = top_idx // vocab_size, top_idx % vocab_size
            seqs = torch.cat([seqs[beam_idx], tok.unsqueeze(1)], dim=1)
            h, c = h[:, beam_idx], c[:, beam_idx]
            ended = tok == self.end_id
            for s, seq in zip(top_scores[ended].tolist(), seqs[ended]):
                done.append((s / (seq.numel() - 1) if length_norm else s, seq[1:-1].tolist()))
            k -= int(ended.sum())
            if k <= 0:
                break
            keep = ~ended
            seqs, scores, h, c = seqs[keep], top_scores[keep], h[:, keep], c[:, keep]
        if not done:  # 到 max_len 還沒結束的候選
            for s, seq in zip(scores.tolist(), seqs):
                done.append((s / (seq.numel() - 1) if length_norm else s, seq[1:].tolist()))
        return max(done, key=lambda t: t[0])[1]


def build_decoder(vocab, feat_dim, embed_dim=256, hidden_dim=512, dropout=0.5) -> CaptionDecoder:
    return CaptionDecoder(
        len(vocab), feat_dim, embed_dim, hidden_dim, dropout,
        pad_id=vocab.stoi[PAD], start_id=vocab.stoi[START], end_id=vocab.stoi[END],
    )


def ids_to_caption(ids, vocab) -> str:
    """字 id -> 句子；遇到 <end> 就停止，並略過 <pad> / <start>。"""
    words = []
    for i in ids:
        w = vocab.itos[i]
        if w == END:
            break
        if w not in (PAD, START):
            words.append(w)
    return " ".join(words)

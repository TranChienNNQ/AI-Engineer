"""Lesson 18: explicit Transformer blocks and a conditional copy task. CPU only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

PAD, BOS, SEP, EOS = 0, 1, 2, 3
SYMBOLS = tuple(range(4, 8))
VOCAB = ['PAD', 'BOS', 'SEP', 'EOS', 'A', 'B', 'C', 'D']
IGNORE = -100


def causal_blocked(length, device=None):
    """Bool True means blocked for nn.MultiheadAttention / nn.Transformer."""
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


def sinusoidal(length, dim, dtype=torch.float32):
    if dim < 2 or dim % 2:
        raise ValueError('This teaching implementation requires even dim >= 2')
    pos = torch.arange(length, dtype=dtype)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2, dtype=dtype) * (-math.log(10000) / dim))
    pe = torch.zeros(length, dim, dtype=dtype)
    pe[:, 0::2], pe[:, 1::2] = (pos * frequency).sin(), (pos * frequency).cos()
    return pe


class Block(nn.Module):
    """Self-attention block; pre/post-LN explicit. All sublayer dropouts are zero."""
    def __init__(self, dim, heads, ff, pre_norm=True, eps=1e-5):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=eps), nn.LayerNorm(dim, eps=eps)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.ReLU(), nn.Linear(ff, dim))
        self.pre_norm = pre_norm

    def forward(self, x, blocked=None, padding=None):
        if self.pre_norm:
            z = self.norm1(x)
            x = x + self.attn(z, z, z, attn_mask=blocked, key_padding_mask=padding,
                              need_weights=False)[0]
            return x + self.ff(self.norm2(x))
        x = self.norm1(x + self.attn(x, x, x, attn_mask=blocked,
                                    key_padding_mask=padding, need_weights=False)[0])
        return self.norm2(x + self.ff(x))


def hand_demo():
    """Uniform attention, two tokens, post-LN. eps=0 ONLY for exact hand math."""
    x = torch.tensor([[[1., 3.], [2., 0.]]], dtype=torch.float64)
    block = Block(2, 1, 3, pre_norm=False, eps=0.).double()
    with torch.no_grad():
        block.attn.in_proj_weight.zero_()
        block.attn.in_proj_weight[4:].copy_(torch.eye(2))
        block.attn.in_proj_bias.zero_()
        block.attn.out_proj.weight.copy_(torch.eye(2))
        block.attn.out_proj.bias.zero_()
        block.ff[0].weight.copy_(torch.tensor([[1., 0.], [-1., 1.], [0., 1.]]))
        block.ff[0].bias.zero_()
        block.ff[2].weight.copy_(torch.tensor([[1., 0., 1.], [0., 1., -1.]]))
        block.ff[2].bias.zero_()
    ln = lambda z: (z - z.mean(-1, keepdim=True)) / z.var(-1, correction=0, keepdim=True).sqrt()
    a = torch.full((1, 2, 2), .5, dtype=torch.float64)
    attention = a @ x
    residual = x + attention
    normalized = ln(residual)
    hidden = F.relu(normalized @ block.ff[0].weight.T)
    feedforward = hidden @ block.ff[2].weight.T
    output = ln(normalized + feedforward)
    torch.testing.assert_close(output, block(x))
    torch.testing.assert_close(output, torch.tensor([[[-1., 1.], [1., -1.]]], dtype=x.dtype))
    result = dict(input=x.tolist(), attention=a.tolist(), attention_output=attention.tolist(),
                  residual=residual.tolist(), norm1=normalized.tolist(), hidden=hidden.tolist(),
                  ff=feedforward.tolist(), output=output.tolist())
    print(json.dumps(result, indent=2))
    return result


def block_demo():
    torch.manual_seed(18)
    reports = {}
    for pre in (False, True):
        custom = Block(8, 2, 16, pre_norm=pre).double()
        reference = nn.TransformerEncoderLayer(8, 2, 16, dropout=0., activation='relu',
                    batch_first=True, norm_first=pre).double()
        with torch.no_grad():
            reference.self_attn.load_state_dict(custom.attn.state_dict())
            reference.linear1.load_state_dict(custom.ff[0].state_dict())
            reference.linear2.load_state_dict(custom.ff[2].state_dict())
            reference.norm1.load_state_dict(custom.norm1.state_dict())
            reference.norm2.load_state_dict(custom.norm2.state_dict())
        # Train mode avoids optimized eval path; dropout is zero in both modules.
        x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        padding = torch.tensor([[False, False, False, False], [False, False, True, True]])
        y = custom(x, causal_blocked(4), padding)
        yr = reference(xr, src_mask=causal_blocked(4), src_key_padding_mask=padding)
        torch.testing.assert_close(y, yr)
        probe = torch.randn_like(y)
        (y * probe).sum().backward()
        (yr * probe).sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        pairs = [(custom.attn, reference.self_attn), (custom.norm1, reference.norm1),
                 (custom.norm2, reference.norm2), (custom.ff[0], reference.linear1),
                 (custom.ff[2], reference.linear2)]
        for left, right in pairs:
            for lp, rp in zip(left.parameters(), right.parameters()):
                torch.testing.assert_close(lp.grad, rp.grad)
        reports['pre' if pre else 'post'] = dict(max_error=(y - yr).abs().max().item(),
                                               parameters=sum(p.numel() for p in custom.parameters()))
    print(json.dumps(reports, indent=2))
    return reports


def position_demo():
    torch.manual_seed(18)
    block = Block(4, 2, 8).double().eval()
    x = torch.randn(1, 4, 4, dtype=torch.float64)
    permutation = [2, 0, 3, 1]
    pe = sinusoidal(4, 4, dtype=x.dtype)
    with torch.no_grad():
        error = (block(x[:, permutation]) - block(x)[:, permutation]).abs().max().item()
        fixed_pe_error = (block(x[:, permutation] + pe) - block(x + pe)[:, permutation]).abs().max().item()
    assert error < 1e-12 and fixed_pe_error > 1e-3
    result = dict(pe=pe.tolist(), without_position_error=error, fixed_position_error=fixed_pe_error)
    print(json.dumps(result, indent=2))
    return result


def masks_demo():
    torch.manual_seed(18)
    block = Block(8, 2, 16).double().eval()
    x = torch.randn(1, 5, 8, dtype=torch.float64, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 100
    mask = causal_blocked(5)
    y = block(x, mask)
    changed_y = block(changed, mask)
    prefix_error = (y[:, :3] - changed_y[:, :3]).abs().max().item()
    torch.testing.assert_close(y[:, :3], changed_y[:, :3])
    y[:, :3].square().sum().backward()
    torch.testing.assert_close(x.grad[:, 3:], torch.zeros_like(x.grad[:, 3:]))
    leak = (block(x.detach())[:, :3] - block(changed)[:, :3]).abs().max().item()
    assert leak > 1e-4
    padding = torch.tensor([[False, False, False, True, True]])
    torch.testing.assert_close(block(x.detach(), mask, padding)[:, :3], block(changed, mask, padding)[:, :3])
    # LayerNorm(dim) has no batch/time coupling; FFN uses the same maps at every token.
    ln = nn.LayerNorm(8).double()
    torch.testing.assert_close(ln(x.detach())[:, :2], ln(x.detach()[:, :2]))
    result = dict(causal_prefix_error=prefix_error, future_gradient_norm=x.grad[:, 3:].norm().item(),
                  noncausal_prefix_change=leak, padded_query_norm=block(x.detach(), mask, padding)[:, 3:].norm().item())
    print(json.dumps(result, indent=2))
    return result


def seq2seq_demo():
    """One full encoder-decoder forward; tgt is SHIFTED before calling model."""
    torch.manual_seed(18)
    embedding = nn.Embedding(8, 8, padding_idx=PAD).double()
    model = nn.Transformer(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
                           dim_feedforward=16, dropout=0., batch_first=True).double()
    head = nn.Linear(8, 8).double()
    src = torch.tensor([[4, 5, EOS, PAD], [6, EOS, PAD, PAD]])
    target = torch.tensor([[BOS, 4, 5, EOS], [BOS, 6, EOS, PAD]])
    tgt, labels = target[:, :-1], target[:, 1:]
    src_pad, tgt_pad = src.eq(PAD), tgt.eq(PAD)
    # Both calls explicitly pass memory_key_padding_mask: source PAD must be hidden from cross-attention too.
    def forward(src_features, tgt_features):
        return head(model(src_features, tgt_features, tgt_mask=causal_blocked(tgt.shape[1]),
                          src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad,
                          memory_key_padding_mask=src_pad))
    se = embedding(src) + sinusoidal(4, 8, dtype=torch.float64)
    te = embedding(tgt) + sinusoidal(3, 8, dtype=torch.float64)
    logits = forward(se, te)
    loss = F.cross_entropy(logits.reshape(-1, 8), labels.reshape(-1), ignore_index=PAD)
    altered = se.detach().clone()
    altered[src_pad] += torch.randn_like(altered[src_pad]) * 100
    torch.testing.assert_close(logits, forward(altered, te))
    future = te.detach().clone()
    future[:, 2] += torch.randn_like(future[:, 2]) * 100
    torch.testing.assert_close(logits[:, :2], forward(se, future)[:, :2])
    loss.backward()
    result = dict(logits_shape=list(logits.shape), supervised_tokens=int(labels.ne(PAD).sum()),
                  loss=loss.item(), target_input=tgt.tolist(), labels=labels.tolist())
    print(json.dumps(result, indent=2))
    return result


@dataclass
class Config:
    dim: int = 32
    heads: int = 4
    ff: int = 64
    layers: int = 2
    max_len: int = 32
    epochs: int = 60
    batch_size: int = 32
    lr: float = .003
    seed: int = 2026
    data_seed: int = 18


class TinyTransformer(nn.Module):
    """Decoder-only: causal self-attention, pre-LN, final LN, untied output head."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(8, cfg.dim, padding_idx=PAD)
        self.register_buffer('pe', sinusoidal(cfg.max_len, cfg.dim))
        # Independent initialization per layer; do not duplicate one module object.
        self.blocks = nn.ModuleList([Block(cfg.dim, cfg.heads, cfg.ff) for _ in range(cfg.layers)])
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, 8)

    def forward(self, ids):
        if ids.ndim != 2 or ids.shape[1] == 0 or ids.shape[1] > self.cfg.max_len:
            raise ValueError('Expected nonempty [B,T] within max_len')
        # This lab accepts only BOS-started right-padded sequences; every query can read BOS.
        if not ids[:, 0].eq(BOS).all():
            raise ValueError('Every row must start with BOS (no left padding)')
        padding = ids.eq(PAD)
        if ((padding.cumsum(1) > 0) & ~padding).any():
            raise ValueError('Only right padding is supported')
        x = self.embedding(ids) * math.sqrt(self.cfg.dim) + self.pe[:ids.shape[1]]
        blocked = causal_blocked(ids.shape[1], ids.device)
        for block in self.blocks:
            x = block(x, blocked, padding)
        return self.head(self.final_norm(x))


def make_records(seed=18):
    # Uniform over all unique source strings of lengths 3..6, then split once by source identity.
    pool = [list(s) for length in range(3, 7) for s in itertools.product(SYMBOLS, repeat=length)]
    random.Random(seed).shuffle(pool)
    return dict(train=pool[:512], validation=pool[512:640], test=pool[640:768])


class CopyDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        source = self.records[index]
        sequence = [BOS] + source + [SEP] + source + [EOS]
        x, y = torch.tensor(sequence[:-1]), torch.tensor(sequence[1:])
        y[:len(source) + 1] = IGNORE  # Supervise copy/EOS only, not random source or SEP.
        return x, y


def collate(batch):
    length = max(len(x) for x, _ in batch)
    xs = torch.full((len(batch), length), PAD, dtype=torch.long)
    ys = torch.full_like(xs, IGNORE)
    for i, (x, y) in enumerate(batch):
        xs[i, :len(x)], ys[i, :len(y)] = x, y
    return xs, ys


def loader(records, cfg, shuffle=False):
    return DataLoader(CopyDataset(records), batch_size=cfg.batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(cfg.seed), collate_fn=collate, num_workers=0)


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    loss_sum = correct = tokens = 0
    for x, y in batches:
        logits = model(x)
        valid = y.ne(IGNORE)
        loss_sum += F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE, reduction='sum').item()
        correct += int((logits.argmax(-1)[valid] == y[valid]).sum())
        tokens += int(valid.sum())
    return dict(ce=loss_sum / tokens, token_accuracy=correct / tokens, tokens=tokens)


@torch.no_grad()
def generate(model, source, max_new=None):
    if not 1 <= len(source) <= 6 or any(s not in SYMBOLS for s in source):
        raise ValueError('Expected 1..6 symbols in IDs 4..7; trained lengths are 3..6')
    steps = len(source) + 1 if max_new is None else max_new
    if steps < 1 or steps + len(source) + 1 > model.cfg.max_len:
        raise ValueError('Generation would exceed positional buffer')
    model.eval()
    ids = torch.tensor([[BOS] + list(source) + [SEP]])
    result = []
    for _ in range(steps):
        # No KV cache: recompute full prefix at each step, inspect last query only.
        token = int(model(ids)[0, -1].argmax())
        result.append(token)
        # An unexpected structural token is an invalid answer; stop and count as a failure.
        if token in (PAD, BOS, SEP, EOS):
            break
        ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    return result


def greedy_metrics(model, records):
    predictions = [generate(model, source) for source in records]
    correct = sum(pred == source + [EOS] for source, pred in zip(records, predictions))
    return dict(exact_match=correct / len(records), correct=correct, samples=len(records))


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = TinyTransformer(Config(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint


def train(out: Path, cfg: Config):
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    records = make_records(cfg.data_seed)
    data_text = json.dumps(records, ensure_ascii=False, sort_keys=True)
    data_hash = hashlib.sha256(data_text.encode()).hexdigest()
    (out / 'data.json').write_text(data_text + '\n')
    (out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2) + '\n')
    model = TinyTransformer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    training = loader(records['train'], cfg, True)
    validation = loader(records['validation'], cfg)
    rows, best = [], float('inf')
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = tokens = 0
        for x, y in training:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            count = int(y.ne(IGNORE).sum())
            loss_sum += loss.item() * count
            tokens += count
        val = evaluate(model, validation)
        rows.append(dict(epoch=epoch, steps=epoch * len(training), train_ce=loss_sum / tokens,
                         val_ce=val['ce'], val_token_accuracy=val['token_accuracy']))
        if val['ce'] < best:
            best = val['ce']
            torch.save(dict(model=model.state_dict(), config=asdict(cfg), epoch=epoch,
                            data_hash=data_hash, vocabulary=VOCAB), out / 'best.pt')
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} train_ce={loss_sum/tokens:.4f} val_ce={val['ce']:.4f} val_acc={val['token_accuracy']:.3f}", flush=True)
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    model, checkpoint = load_model(out / 'best.pt')
    val = evaluate(model, validation)
    val.update(greedy_metrics(model, records['validation']))
    test = evaluate(model, loader(records['test'], cfg))  # Only after best validation checkpoint selected.
    test.update(greedy_metrics(model, records['test']))
    # Train-derived constant token baseline, including EOS in the supervised target domain.
    targets = [token for source in records['train'] for token in source + [EOS]]
    majority = max(sorted(set(targets)), key=targets.count)
    baselines = {}
    for split in ('validation', 'test'):
        labels = [token for source in records[split] for token in source + [EOS]]
        baselines[split] = dict(constant_token_id=majority, token_accuracy=labels.count(majority)/len(labels),
                                constant_greedy_exact=0., rule_copy_exact=1.)
    probe = records['test'][0]
    with torch.no_grad():
        inp, _ = CopyDataset([probe])[0]
        before = model(inp[None])
        restored, _ = load_model(out / 'best.pt')
        torch.testing.assert_close(before, restored(inp[None]), rtol=0, atol=0)
    report = dict(parameters=sum(p.numel() for p in model.parameters()), best_epoch=checkpoint['epoch'],
                  updates=cfg.epochs*len(training), data_hash=data_hash, validation=val, test=test,
                  baselines=baselines, probe=dict(source=probe, expected=probe+[EOS], generated=generate(model, probe)),
                  checkpoint_roundtrip=True)
    metadata = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    device='CPU', dtype='float32', threads=1, platform=platform.platform())
    for name, obj in [('report', report), ('metadata', metadata)]:
        (out / f'{name}.json').write_text(json.dumps(obj, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['hand', 'block', 'position', 'masks', 'seq2seq', 'train', 'all'], default='hand')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(1)
    demos = dict(hand=hand_demo, block=block_demo, position=position_demo, masks=masks_demo, seq2seq=seq2seq_demo)
    for name, fn in demos.items():
        if args.mode in (name, 'all'):
            print('\nMODE:', name)
            fn()
    if args.mode in ('train', 'all'):
        out = args.out or Path(tempfile.mkdtemp(prefix='lesson18-')) / 'run'
        train(out, Config(epochs=args.epochs))
        print('Output:', out)


if __name__ == '__main__':
    main()

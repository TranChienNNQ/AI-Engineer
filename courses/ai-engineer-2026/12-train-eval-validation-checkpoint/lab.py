"""CPU training lifecycle; synthetic regression, no external data or tracking service."""
import argparse
import copy
import csv
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

DTYPE = torch.float64

@dataclass(frozen=True)
class Config:
    seed: int = 2026
    data_seed: int = 17
    lr: float = 0.03
    momentum: float = 0.9
    batch_size: int = 16
    hidden: int = 16
    dropout: float = 0.15
    epochs: int = 60
    patience: int = 15
    min_delta: float = 0.0


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def make_data(config):
    """IDs are row IDs, not user groups. One fixed, disjoint synthetic split."""
    generator = torch.Generator().manual_seed(config.data_seed)
    X = 2*torch.rand((90, 1), generator=generator, dtype=DTYPE)-1
    y = torch.sin(3*X)+0.3*X+0.2*torch.randn((90, 1), generator=generator, dtype=DTYPE)
    ids = torch.randperm(90, generator=generator)
    split = {'train': ids[:48].tolist(), 'validation': ids[48:69].tolist(), 'test': ids[69:].tolist()}
    datasets = {key: TensorDataset(X[index], y[index]) for key, index in split.items()}
    raw = X.numpy().tobytes()+y.numpy().tobytes()+json.dumps(split, sort_keys=True).encode()
    return datasets, split, hashlib.sha256(raw).hexdigest()


def make_loader(dataset, config, epoch=0, training=False):
    # Reconstruct epoch-specific ordering; no persistent sampler state to restore.
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training,
                      drop_last=False, num_workers=0,
                      generator=torch.Generator().manual_seed(config.seed+epoch))


def make_model(config):
    return nn.Sequential(nn.Linear(1, config.hidden), nn.Tanh(),
                         nn.Dropout(config.dropout), nn.Linear(config.hidden, 1)).to(dtype=DTYPE, device='cpu')


def train_epoch(model, loader, optimizer):
    model.train()
    total, count, steps = 0.0, 0, 0
    for X, y in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(X)
        if prediction.shape != y.shape:
            raise ValueError('Prediction and target must have equal shape.')
        loss = (prediction-y).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss; inspect input and learning rate.')
        loss.backward()
        optimizer.step()
        total += loss.item()*len(y)
        count += len(y)
        steps += 1
    if not count:
        raise ValueError('Empty train loader.')
    return total/count, steps


@torch.no_grad()
def evaluate(model, loader):
    """One scalar target per sample: sum squared errors / actual sample count."""
    model.eval()
    total, count = 0.0, 0
    for X, y in loader:
        prediction = model(X)
        if prediction.shape != y.shape:
            raise ValueError('Prediction and target must have equal shape.')
        total += (prediction-y).square().sum().item()
        count += len(y)
    if not count:
        raise ValueError('Empty evaluation loader.')
    metric = total/count
    if not math.isfinite(metric):
        raise FloatingPointError('Non-finite evaluation metric.')
    return metric


def selection_trace(values, patience=2, min_delta=0.01):
    """Hand example, LOWER is better; strict improvement, ties do not reset."""
    best, bad, best_epoch = math.inf, 0, 0
    rows = []
    for epoch, value in enumerate(values, 1):
        improved = value < best-min_delta
        if improved:
            best, best_epoch, bad = value, epoch, 0
        else:
            bad += 1
        stop = bad >= patience
        rows.append((epoch, value, improved, best_epoch, bad, stop))
        if stop:
            break
    return rows


def atomic_save(obj, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)  # Same-directory rename; not a full durable storage protocol.


def read_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def save_best(out, config, state):
    atomic_save({'schema': 1, 'config': asdict(config), 'epoch': state['best_epoch'],
                 'val_mse': state['best_val'], 'data_sha256': state['data_sha256'],
                 'model': state['best_model']}, Path(out)/'best.pt')


def fit(out, config=Config(), resume=None, until=None):
    """Resume only at an epoch boundary; new output directory, never overwrite a run."""
    if not (config.lr > 0 and config.epochs > 0 and config.batch_size > 0 and
            config.patience > 0 and config.min_delta >= 0):
        raise ValueError('Invalid training configuration.')
    end = config.epochs if until is None else until
    if not 1 <= end <= config.epochs:
        raise ValueError('until must be in [1, epochs].')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    data, split, fingerprint = make_data(config)
    model = make_model(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    state = {'epoch': 0, 'global_step': 0, 'best_val': math.inf, 'best_epoch': 0,
             'bad_epochs': 0, 'history': [], 'data_sha256': fingerprint, 'stopped': False}
    if resume is not None:
        state = read_checkpoint(resume)
        if state['schema'] != 1 or state['config'] != asdict(config):
            raise ValueError('Resume requires the identical config and schema.')
        if state['data_sha256'] != fingerprint:
            raise ValueError('Data/split changed.')
        if state['stopped'] or state['epoch'] >= end:
            raise ValueError('Checkpoint already stopped or reached requested end.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])  # After model initialization.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', asdict(config))
    write_json(out/'split.json', {'ids': split, 'data_sha256': fingerprint})
    write_json(out/'metadata.json', {'python': platform.python_version(), 'torch': str(torch.__version__),
               'device': 'cpu', 'dtype': 'float64', 'threads': 1,
               'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'parent_checkpoint': str(resume) if resume is not None else None})
    fields = ['epoch', 'global_step', 'lr', 'train_online_mse', 'train_eval_mse', 'val_mse',
              'best_epoch', 'bad_epochs']
    with (out/'metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(state['history'])  # A resumed run includes its prefix history.
        for epoch in range(state['epoch']+1, end+1):
            online, steps = train_epoch(model, make_loader(data['train'], config, epoch, True), optimizer)
            train_mse = evaluate(model, make_loader(data['train'], config))
            val_mse = evaluate(model, make_loader(data['validation'], config))
            state['global_step'] += steps
            if val_mse < state['best_val']-config.min_delta:
                state.update(best_val=val_mse, best_epoch=epoch, bad_epochs=0,
                             best_model=copy.deepcopy(model.state_dict()))
                save_best(out, config, state)
            else:
                state['bad_epochs'] += 1
            row = dict(epoch=epoch, global_step=state['global_step'], lr=optimizer.param_groups[0]['lr'],
                       train_online_mse=online, train_eval_mse=train_mse, val_mse=val_mse,
                       best_epoch=state['best_epoch'], bad_epochs=state['bad_epochs'])
            state['history'].append(row)
            writer.writerow(row)
            file.flush()
            state.update(schema=1, config=asdict(config), epoch=epoch,
                         stopped=state['bad_epochs'] >= config.patience,
                         model=model.state_dict(), optimizer=optimizer.state_dict(),
                         torch_rng=torch.get_rng_state())
            atomic_save(state, out/'last.pt')
            if state['stopped']:
                break
    # A self-contained resumed run also gets the best artifact from the saved prefix.
    save_best(out, config, state)
    summary = {k: state[k] for k in ['epoch', 'global_step', 'best_epoch', 'best_val', 'stopped']}
    write_json(out/'summary.json', summary)
    return summary


def load_best(path):
    checkpoint = read_checkpoint(path)
    model = make_model(Config(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, checkpoint


def study(out):
    """Two predeclared learning rates; select on validation, score test only once."""
    out = Path(out)
    candidates = []
    for name, lr in [('lr-003', 0.03), ('lr-0005', 0.005)]:
        config = Config(lr=lr)
        result = fit(out/name, config)
        candidates.append({'run': name, 'lr': lr, **result})
    winner = min(candidates, key=lambda r: r['best_val'])  # First candidate wins an exact tie.
    model, checkpoint = load_best(out/winner['run']/'best.pt')
    data, _, fingerprint = make_data(Config(**checkpoint['config']))
    assert fingerprint == checkpoint['data_sha256']
    test_mse = evaluate(model, make_loader(data['test'], Config(**checkpoint['config'])))
    report = {'candidates': candidates, 'selected': winner['run'], 'test_mse': test_mse,
              'test_n': len(data['test']), 'selection_metric': 'minimum validation MSE'}
    write_json(out/'study.json', report)
    print(json.dumps(report, indent=2))
    return report


def resume_demo(out):
    out = Path(out)
    config = Config(epochs=12, patience=100)
    full = fit(out/'full', config)
    fit(out/'part', config, until=5)
    continued = fit(out/'continued', config, resume=out/'part'/'last.pt')
    a, b = read_checkpoint(out/'full'/'last.pt'), read_checkpoint(out/'continued'/'last.pt')
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['history'] == b['history']
    assert torch.equal(a['torch_rng'], b['torch_rng'])
    # Show that loading weights alone loses SGD's momentum buffers.
    wrong = make_model(config); wrong.load_state_dict(read_checkpoint(out/'part'/'last.pt')['model'])
    wrong_optimizer = torch.optim.SGD(wrong.parameters(), lr=config.lr, momentum=config.momentum)
    torch.set_rng_state(read_checkpoint(out/'part'/'last.pt')['torch_rng'])
    for epoch in range(6, 13):
        data, _, _ = make_data(config)
        train_epoch(wrong, make_loader(data['train'], config, epoch, True), wrong_optimizer)
    difference = max((p-wrong.state_dict()[k]).abs().max().item() for k,p in a['model'].items())
    assert difference > 1e-8
    restored, _ = load_best(out/'full'/'best.pt')
    original = make_model(config);original.load_state_dict(a['best_model']);original.eval()
    probe = torch.tensor([[-0.5], [0.0], [0.5]], dtype=DTYPE)
    with torch.no_grad():
        torch.testing.assert_close(original(probe), restored(probe), rtol=0, atol=0)
    result = {'exact_resume': True, 'prediction_roundtrip': True,
              'weights_only_resume_parameter_max_difference': difference,
              'full': full, 'continued': continued}
    write_json(out/'resume-check.json', result)
    print(json.dumps(result, indent=2))
    return result


def hand_demo():
    model = nn.Linear(1,1).to(dtype=DTYPE)
    with torch.no_grad(): model.weight.zero_();model.bias.zero_()
    X = torch.tensor([[0.], [1.]], dtype=DTYPE);y=2*X+1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = (model(X)-y).square().mean();loss.backward()
    assert model.weight.grad.item()==-3 and model.bias.grad.item()==-4
    optimizer.step()
    mse = evaluate(model, DataLoader(TensorDataset(X,y), batch_size=1))
    assert abs(mse-2.825)<1e-12
    print('Initial loss=5; dw=-3; db=-4; after SGD: w=0.3,b=0.4; fixed-model MSE=',mse)
    print('epoch,val,improved,best_epoch,bad_epochs,stop')
    for row in selection_trace([.8,.6,.595,.61,.58], patience=2,min_delta=.01): print(row)


def modes_demo():
    torch.manual_seed(7)
    layer=nn.Dropout(0.5)
    x=torch.ones(32,dtype=DTYPE,requires_grad=True)
    layer.train()
    with torch.no_grad(): training_output=layer(x)
    layer.eval();eval_output=layer(x)
    assert set(training_output.tolist())=={0.0,2.0}
    assert torch.equal(eval_output,x) and eval_output.requires_grad
    print('Dropout train + no_grad: values {0,2}; eval: all 1, still requires_grad=True')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['hand','modes','study','resume','all'],default='all')
    parser.add_argument('--out',type=Path, help='New directory; default is a fresh temporary location.')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode in ('hand','all'): hand_demo()
    if args.mode in ('modes','all'): modes_demo()
    if args.mode in ('study','resume','all'):
        out=args.out if args.out is not None else Path(tempfile.mkdtemp(prefix='lesson12-'))/'runs'
        if out.exists(): raise FileExistsError('Choose a new output directory.')
        if args.mode in ('study','all'): study(out/'study')
        if args.mode in ('resume','all'): resume_demo(out/'resume')
        print('Artifacts:',out.resolve())

if __name__=='__main__': main()

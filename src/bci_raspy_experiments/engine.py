"""Trial-matched optimization, fixed-schedule refit, and exact epoch resume."""
import copy
import os
import random
from pathlib import Path
import numpy as np
import torch
from .common import digest, save_json
from .models import build_model, describe
from .windows import loader


def seed_all(seed, threads=4):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def state_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_torch(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def load_torch(path, device='cpu'):
    # Only local, self-produced checkpoints; never load untrusted pickles.
    return torch.load(path, map_location=device, weights_only=False)


def loss_for(logits, labels, objective, name, weights):
    if objective == 'mil':
        values = logits.mean(dim=1)
        target = labels
    else:
        values = logits.flatten(0, 1)
        target = labels[:, None].expand(-1, logits.shape[1]).flatten()
    if name == 'ce':
        return torch.nn.functional.cross_entropy(values, target)
    onehot = torch.nn.functional.one_hot(target, 4).float()
    errors = (values - onehot).square().mean(-1)
    weight = weights[target]
    return 100. * (errors * weight / weight.mean()).mean()


def forward_bags(model, x):
    b, w = x.shape[:2]
    return model(x.flatten(0, 1)).reshape(b, w, 4)


def validation(model, bags, cfg, profile, device, weights):
    total_loss, count, correct, instances = 0., 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader(bags, cfg, profile):
            x, y = batch['x'].to(device), batch['y'].to(device)
            z = forward_bags(model, x)
            loss = loss_for(z, y, cfg['objective'], cfg['loss'], weights)
            total_loss += float(loss) * len(y)
            count += len(y)
            correct += int((z.argmax(-1) == y[:, None]).sum())
            instances += z.shape[0] * z.shape[1]
    return total_loss / count, correct / instances


def fit(train, validation_data, cfg, profile, directory, resume=False, fixed_schedule=None, stop_after=None):
    """Validation is absent during refit. This function never accepts a test loader
    in strict selection/refit; orchestration enforces its data identity.

    stop_after is a test hook to exercise interruption at an epoch boundary.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = digest([cfg, train.ids, None if validation_data is None else validation_data.ids, fixed_schedule,
                        profile['device'], profile['workers'], profile['threads']])
    path = directory / 'last.pt'
    if path.exists() and not resume:
        raise FileExistsError(f'Use --resume for {directory}')
    seed_all(cfg['seed'], profile['threads'])
    device = torch.device(profile['device'])
    sample = train[0]['x']
    channels, samples = sample.shape[1:3]
    model = build_model(cfg['model'], channels, samples).to(device)
    shape_info = describe(model, channels, samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=.1)
    counts = np.bincount([train.rows[i]['label'] for i in train.ids], minlength=4)
    if np.any(counts == 0):
        raise ValueError('Missing training class')
    weights = torch.tensor(counts.mean()/counts, dtype=torch.float32, device=device)
    history, schedule = [], []
    best, best_index, best_loss, stale, start = None, 0, float('inf'), 0, 0
    if resume and path.exists():
        checkpoint = load_torch(path, device)
        if checkpoint['signature'] != signature:
            raise ValueError('Resume config/split/runtime mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        history, schedule = checkpoint['history'], checkpoint['schedule']
        best, best_index, best_loss, stale = (checkpoint[k] for k in ('best', 'best_index', 'best_loss', 'stale'))
        start = checkpoint['epoch'] + 1
        restore_rng(checkpoint['rng'])
        if checkpoint['complete']:
            model.load_state_dict(best if fixed_schedule is None else checkpoint['model'])
            return model, dict(history=history, schedule=schedule[:best_index+1] if fixed_schedule is None else schedule,
                               best_epoch=best_index, shape=shape_info)
    max_epochs = len(fixed_schedule) if fixed_schedule is not None else cfg['train']['max_epochs']
    for epoch in range(start, max_epochs):
        if fixed_schedule is not None:
            for group in optimizer.param_groups:
                group['lr'] = fixed_schedule[epoch]
        lr = optimizer.param_groups[0]['lr']
        schedule.append(lr)
        model.train()
        summed, count, updates, instances = 0., 0, 0, 0
        for batch in loader(train, cfg, profile, epoch, shuffle=True):
            x, y = batch['x'].to(device), batch['y'].to(device)
            optimizer.zero_grad()
            z = forward_bags(model, x)
            loss = loss_for(z, y, cfg['objective'], cfg['loss'], weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            summed += float(loss.detach()) * len(y)
            count += len(y)
            updates += 1
            instances += z.shape[0] * z.shape[1]
        record = dict(epoch=epoch, train_loss=summed/count, lr=lr, updates=updates, instances=instances)
        if validation_data is not None:
            val_loss, accuracy = validation(model, validation_data, cfg, profile, device, weights)
            record.update(selection_loss=val_loss, selection_window_accuracy=accuracy)
            scheduler.step(accuracy)
            if val_loss < best_loss:
                best_loss, best_index, stale = val_loss, epoch, 0
                best = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        else:
            best = copy.deepcopy(model.state_dict())
            best_index, best_loss = epoch, summed/count
        history.append(record)
        complete = epoch + 1 == max_epochs or (validation_data is not None and epoch > 5 and stale >= cfg['train']['patience'])
        checkpoint = dict(signature=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scheduler=scheduler.state_dict(), rng=state_rng(), epoch=epoch, history=history,
                          schedule=schedule, best=best, best_index=best_index, best_loss=best_loss,
                          stale=stale, complete=complete, sampler=dict(seed=cfg['seed'], next_epoch=epoch+1))
        save_torch(path, checkpoint)
        save_json(directory / 'history.json', history)
        print(f'{directory.parent.name}/{directory.name} epoch={epoch+1} loss={summed/count:.5g}', flush=True)
        if stop_after is not None and epoch + 1 >= stop_after and not complete:
            raise InterruptedError('Requested test interruption')
        if complete:
            break
    model.load_state_dict(best)
    return model, dict(history=history, schedule=schedule[:best_index+1] if fixed_schedule is None else schedule,
                       best_epoch=best_index, shape=shape_info)

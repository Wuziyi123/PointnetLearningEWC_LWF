import os

import torch
from torch import nn
import numpy as np

from core import PointNetClassifier, EarlyStopper, train_path
from core.ewc_loss import EWCLoss
from core.raport_generator import log_msg, RaportGenerator
from core.dataset import test_transforms, default_transforms
from torch.utils.data import DataLoader


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_datasets(dataset, tasks, sampling, kind, transforms):
    datasets = {}
    for task, classes in tasks.items():
        data = dataset(sampling=sampling, classes=classes, split=kind, transform=transforms())
        datasets[task] = data
    return datasets


def get_fisher_optpars(params):
    fisher_dict = {}
    optpar_dict = {}
    for name, param in params:
        if param.grad is not None:
            fisher_dict[name] = param.grad.data.clone().pow(2).mean(0)
            optpar_dict[name] = param.data.clone()
    return fisher_dict, optpar_dict


def train_model_ewc(
        dataset,
        tasks,
        sampling,
        z_size,
        batch_norm,
        verbose,
        max_epochs,
        batch_size,
        lr,
        patience,
        alpha):
    log_msg(f'Start EWC process, device: {DEVICE}')
    log_msg(f'Tasks: {tasks}')

    # setup val dataset
    train_datasets = setup_datasets(dataset, tasks, sampling, 'train', default_transforms)
    val_datasets = setup_datasets(dataset, tasks, sampling, 'test', test_transforms)
    model = PointNetClassifier(sampling, z_size, batch_norm)
    for task in tasks.keys():
        model.add_final_layer(task, len(tasks[task]))
    log_msg(f'Model setup correctly, sampling: {sampling}, z_size: {z_size}, batch_size: {batch_size}')
    # setup raport generator
    task_names = list(tasks.keys())
    raport = RaportGenerator(task_names, train_path)

    # setup save path
    save_path = os.path.join(train_path, 'checkpoints')

    # train model on task 1
    task1 = task_names[0]
    best_idx, fisher, optpar = train_model_first_task(
        task1,
        tasks,
        model,
        max_epochs,
        lr,
        patience,
        verbose,
        batch_size,
        train_datasets[task1],
        val_datasets[task1],
        save_path
    )
    train_accuracy, test_accuracy = validate_model_all_tasks(
        task1,
        save_path,
        best_idx,
        tasks,
        model,
        train_datasets,
        val_datasets
    )
    log_msg(f'Validate on all tasks, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}', verbose)
    raport.add_info(train_accuracy, test_accuracy)
    previous_tasks = [task1]
    # train on next tasks
    for task in task_names[1:]:
        best_idx, fisher, optpar = train_new_task(
            previous_tasks,
            task,
            tasks,
            model,
            max_epochs,
            lr,
            patience,
            verbose,
            batch_size,
            train_datasets[task],
            val_datasets[task],
            save_path,
            alpha,
            fisher,
            optpar
        )
        train_accuracy, test_accuracy = validate_model_all_tasks(
            task,
            save_path,
            best_idx,
            tasks,
            model,
            train_datasets,
            val_datasets
        )
        log_msg(f'Validate on all tasks, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}', verbose)
        raport.add_info(train_accuracy, test_accuracy)
        previous_tasks += [task]
    raport.save()



def train_model_first_task(
        task,
        tasks,
        model,
        max_epochs,
        lr,
        patience,
        verbose,
        batch_size,
        train_dataset,
        val_dataset,
        save_path
):
    log_msg(
        f'Train model on first task: {tasks[task]}, training samples: {len(train_dataset)}, test samples: {len(val_dataset)}',
        verbose)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log_msg(f'Start training, epochs: {max_epochs}, patience: {patience}, lr: {lr}')
    early_stopper = EarlyStopper(patience)

    # metrics
    loss_history = []
    val_accuracy_history = []
    model.set_active_keys([task])

    # fisher and optpar
    fishers = []
    optpars = []
    for i in range(max_epochs):
        loss_epoch = 0
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss = criterion(pred[task], y.long())
            loss_epoch += loss.item()
            loss.backward()

            optimizer.step()
        fisher_dict, optpar_dict = get_fisher_optpars(model.named_parameters())
        fishers.append(fisher_dict)
        optpars.append(optpar_dict)
        # calculate metrics
        val_accuracy = accuracy_score(task, model, val_loader)
        mean_loss = loss_epoch / len(train_loader)

        # early stopper
        early_stopper.stop(val_accuracy)

        # append to metrics
        loss_history.append(mean_loss)
        val_accuracy_history.append(val_accuracy)

        # log msg
        log_msg(
            f'Epoch {i + 1}/{max_epochs} loss: {mean_loss}, val_accuracy: {val_accuracy}',
            verbose
        )

        # save model
        path = os.path.join(save_path, f'{task}_epoch_{i + 1}_checkpoint.pth')
        torch.save(model.state_dict(), path)
        log_msg(f'Saved checkpoint to {path}')
        if early_stopper:
            log_msg('Early stopping!', verbose)
            break
    best_idx = np.argmax(val_accuracy_history) + 1
    log_msg(f'Best epoch: {best_idx}')
    return best_idx, optpars[best_idx - 1], fishers[best_idx - 1]


def train_new_task(
        prev_tasks,
        new_task,
        tasks,
        model,
        max_epochs,
        lr,
        patience,
        verbose,
        batch_size,
        train_dataset,
        val_dataset,
        save_path,
        alpha,
        fisher,
        optpar
):
    log_msg(
        f'Train model on task: {tasks[new_task]}, training samples: {len(train_dataset)}, test samples: {len(val_dataset)}',
        verbose)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    log_msg(f'Dataset setup correctly, previous_tasks: {prev_tasks}', verbose)
    criterion = nn.CrossEntropyLoss()
    ewc_criterion = EWCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # setup early stopper
    early_stopper = EarlyStopper(patience)

    # metrics
    loss_history = []
    val_accuracy_history = []

    # params
    fishers = []
    optpars = []
    log_msg(f'Start training, epochs: {max_epochs}, patience: {patience}, lr: {lr}', verbose)
    for i in range(max_epochs):
        loss_epoch = 0
        model.train()
        model.set_active_keys(prev_tasks + [new_task])
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred[new_task], y.long())

            loss += ewc_criterion(fisher, optpar, model.named_parameters()) * alpha / len(prev_tasks)
            loss_epoch += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        fisher_dict, optpar_dict = get_fisher_optpars(model.named_parameters())
        fishers.append(fisher_dict)
        optpars.append(optpar_dict)
        # calculate metrics
        mean_loss = loss_epoch / len(train_loader)
        val_accuracy = accuracy_score(new_task, model, val_loader)
        loss_history.append(mean_loss)
        val_accuracy_history.append(val_accuracy)
        # early stopping
        early_stopper.stop(val_accuracy)
        # log epoch msg
        log_msg(
            f'Epoch {i + 1}/{max_epochs} loss: {mean_loss}, val_accuracy: {val_accuracy}',
            verbose
        )
        # save model
        path = os.path.join(save_path, f'{new_task}_epoch_{i + 1}_checkpoint.pth')
        torch.save(model.state_dict(), path)
        log_msg(f'Saved checkpoint to {path}')
        if early_stopper:
            log_msg('Early stopping!', verbose)
            break
    best_idx = np.argmax(val_accuracy_history) + 1
    log_msg(f'Best epoch: {best_idx}')
    return best_idx, fishers[best_idx - 1], optpars[best_idx - 1]


def accuracy_score(task, model, loader):
    model.eval()
    correct = total = 0
    model.set_active_keys([task])
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            _, predict = torch.max(pred[task].data, 1)
            total += y.size(0)
            correct += (predict == y).sum().item()
    return correct / total


def load_model_best_path(task, save_path, best_idx, model):
    best_path = os.path.join(save_path, f'{task}_epoch_{best_idx}_checkpoint.pth')
    model.load_state_dict(torch.load(best_path))
    return model


def validate_model_all_tasks(task, save_path, best_idx, tasks, model, train_datasets, val_datasets):
    train_accuracy = {}
    test_accuracy = {}
    best_model = load_model_best_path(task, save_path, best_idx, model)
    for task in tasks.keys():
        train_accuracy[task] = accuracy_score(task, best_model, DataLoader(train_datasets[task]))
        test_accuracy[task] = accuracy_score(task, best_model, DataLoader(val_datasets[task]))
    return train_accuracy, test_accuracy

import torch
import torch.nn
from torchmetrics.classification import Accuracy

def train_step(model, loss_function, optimizer,device, pbar) :

    # transfer model to device and prepare it for training
    model = model.to(device)
    model.train()

    # define statistics
    accuracy = Accuracy(task = "multiclass", num_classes = 10)
    accuracy = accuracy.to(device)
    running_loss = 0.0
    total_batches = len(pbar)

    # iterate over dataset
    for batch_idx, (inputs, targets) in enumerate(pbar) :

        # send to device
        inputs, targets = inputs.to(device), targets.to(device)

        # perform gradient descent
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        # update statistics
        accuracy.update(outputs, targets)
        running_loss += loss.item()
        pbar.set_postfix(loss = loss.item())

    avg_epoch_loss = running_loss / total_batches
    epoch_accuracy = accuracy.compute().item()
    return model, avg_epoch_loss, epoch_accuracy


def evaluate(model, eval_loader, device) :

    # transfer model to device and prepare it for eval
    model = model.to(device)
    model.eval()

    # define statistics
    accuracy = Accuracy(task = "multiclass", num_classes = 10)
    accuracy = accuracy.to(device)

    with torch.no_grad() :

        for inputs, targets in eval_loader :

            # send to device
            inputs,targets = inputs.to(device), targets.to(device)

            # perform inference
            outputs = model(inputs)

            # update metric
            accuracy.update(outputs, targets)

    eval_accuracy = accuracy.compute().item()
    return eval_accuracy




    

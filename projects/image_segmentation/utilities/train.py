import torch
import torch.nn


def train_step(model, loss_function, optimizer,
               device, pbar, metric) :

    # transfer model to device and prepare it for training
    model = model.to(device)
    model.train()

    # define statistics
    running_loss = 0.0
    total_batches = len(pbar)

    # iterate over dataset
    for batch_idx, (inputs, targets) in enumerate(pbar) :

        # send to device
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float()
        targets =targets.unsqueeze(1)

        # perform gradient descent
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        # update statistics
        metric.update(outputs, targets)
        running_loss += loss.item()
        pbar.set_postfix(loss = loss.item())


    avg_epoch_loss = running_loss / total_batches
    epoch_metric = metric.compute().item()

    # free up some memory 
    metric.reset()
    if torch.backends.mps.is_available() :
        torch.mps.empty_cache()

    return model, avg_epoch_loss, epoch_metric


def evaluate(model, pbar, loss_function, device, metric) :

    # transfer model to device and prepare it for eval
    model = model.to(device)
    model.eval()
    running_loss = 0
    total_batches = len(pbar)

    with torch.no_grad() :

        for batch_idx, (inputs, targets) in enumerate(pbar):

            # send to device
            inputs,targets = inputs.to(device), targets.to(device)
            targets = targets.float()
            targets = targets.unsqueeze(1)

            # perform inference
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            # update statistics
            metric.update(outputs, targets)
            running_loss += loss.item()
            pbar.set_postfix(loss = loss.item())

    # compute metrics
    val_loss = running_loss / total_batches
    eval_metric = metric.compute().item()

    # free up some memory 
    metric.reset()
    if torch.backends.mps.is_available() :
        torch.mps.empty_cache()

    return val_loss, eval_metric




    

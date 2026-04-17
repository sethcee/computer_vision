# packages
from tqdm import tqdm
from utilities.train_step import train_step, evaluate


def train_model(model, train_loader, val_loader, loss_function, optimizer, device, num_epochs) : 
    
    num_epochs = 5

    for epoch in range(num_epochs) :

        # instantiate training progress bar
        pbar = tqdm(train_loader, desc = f"Epoch {epoch + 1}/{num_epochs}", leave = True)

        # train 1 epoch on training set
        model, avg_epoch_loss, train_accuracy = train_step(model, loss_function, optimizer,device, pbar)

        # evaluate on val test
        
    
    val_accuracy = evaluate(model, val_loader, device)
    return val_accuracy
import gc
import tqdm as tqdm
import torch
import pandas as pd
import numpy as np
import copy


def train_epoch(model, optimizer, data_loader, loss_fn, device, scheduler) :
    running_loss = 0.0
    dataset_size = 0.0
    model.train()
    batches_data = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, batch_data in batches_data:
        input_ids = batch_data['input_ids'].to(device, dtype=torch.long)
        attention_mask = batch_data['attention_mask'].to(device, dtype=torch.long)
        labels = batch_data['labels'].to(device, dtype=torch.float)
        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        # the batch tail could include different size, if size is same ignore it
        batch_size = input_ids.size(0)
        running_loss += (loss.item()*batch_size)
        dataset_size += batch_size
        # update parameters
        loss.backward() # compute gradients in the network
        optimizer.step() # update net weights using the gradients and optimizer
        optimizer.zero_grad() # reset optimizer grads for next iteration
        scheduler.step() # iterate in optimizer scheduler method
    batch_loss = running_loss / dataset_size
    gc.collect()
    return batch_loss


def fit(model, optimizer, train_loader, val_loader, loss_fn, device, epochs, model_path, scheduler, epoch_validator):
    history = []
    loss_val_best = np.inf
    # Epoch loop
    for epoch in range(epochs):
        loss_train = train_epoch(model, optimizer, train_loader, loss_fn, device,
                                 scheduler)
        loss_val = epoch_validator(model, optimizer, val_loader, loss_fn, device)
        epoch_history = {'epoch': epoch,
                         'train_loss:': loss_train,
                         'val_loss': loss_val}
        print(epoch_history)
        # save best model
        history.append(epoch_history)
        if loss_val < loss_val_best:
            loss_val_best = loss_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_path)

    # load best model
    model.load_state_dict(best_model_wts)
    gc.collect()
    return model, pd.DataFrame(history)
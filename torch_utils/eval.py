import gc
import tqdm as tqdm
import torch


def eval(model, data_loader, device):
    model.eval()
    batches_data = tqdm(enumerate(data_loader), total=len(data_loader))
    predictions = []
    targets = []
    for _, batch_data in batches_data:
        gc.collect()
        input_ids = batch_data['input_ids'].to(device, dtype=torch.long)
        attention_mask = batch_data['attention_mask'].to(device, dtype=torch.long)
        labels = batch_data['labels'].to(device, dtype=torch.float)
        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask)
        prediction_batch = torch.sigmoid(logits)
        targets.extend(labels.cpu().detach().numpy().tolist())
        predictions.extend(prediction_batch.cpu().detach().numpy().tolist())
    return predictions, targets


def validate_epoch(model, data_loader, loss_fn, device):
  running_loss = 0.0
  dataset_size = 0.0
  model.eval()
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
  batch_loss = running_loss / dataset_size
  gc.collect()
  return batch_loss
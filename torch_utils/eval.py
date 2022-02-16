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
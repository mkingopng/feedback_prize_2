from deberta_large_functions import *


# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for data in tk0:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        with torch.no_grad():
            y_preds, _ = model(ids, mask)
        y_preds = softmax(y_preds.to('cpu').numpy())
        preds.append(y_preds)
    predictions = np.concatenate(preds)
    return predictions


deberta_predictions = []
test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=CFG.num_workers,
    pin_memory=True,
    drop_last=False
)

deberta_predictions = []

for fold in CFG.trn_fold:
    print("Fold {}".format(fold))
    model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
    state = torch.load(
        CFG.path + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device('cpu')
    )
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    deberta_predictions.append(prediction)
    del model, state, prediction;
    gc.collect()
    torch.cuda.empty_cache()


predictions = np.mean(deberta_predictions, axis=0)

submission['Ineffective'] = predictions[:, 0]
submission['Adequate'] = predictions[:, 1]
submission['Effective'] = predictions[:, 2]

print(submission)

submission.to_csv('submission.csv', index=False)

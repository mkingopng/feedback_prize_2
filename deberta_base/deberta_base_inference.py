from deberta_base_functions import *

# Deberta-base inference

testDataset = TestDataset(CFG, test)
test_loader = DataLoader(
    testDataset,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    batch_size=CFG.batch_size,
    num_workers=CFG.num_workers,
    collate_fn=Collate(CFG.tokenizer, isTrain=False)
)

deberta_predictions = []
for i in CFG.trn_fold:
    model = FeedBackModel(CFG.model)
    model.load_state_dict(
        torch.load('model_zoo/deberta_v3_base/deberta-v3-base_fold' + str(i) +
                   '_best.pth')
    )
    prediction = inference_fn(test_loader, model, device)
    deberta_predictions.append(prediction)
    torch.cuda.empty_cache()
    gc.collect()

deb_adequate = []
deb_effective = []
deb_ineffective = []

for x in deberta_predictions:
    deb_ineffective.append(x[:, 0])
    deb_adequate.append(x[:, 1])
    deb_effective.append(x[:, 2])

deb_ineffective = pd.DataFrame(deb_ineffective).T
deb_adequate = pd.DataFrame(deb_adequate).T
deb_effective = pd.DataFrame(deb_effective).T

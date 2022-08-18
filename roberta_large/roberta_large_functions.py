from roberta_large_CFG import *


# model definition
class callback:
    def __init__(self):
        self.loss = list()
        self.model = list()

    def put(self, model, loss):
        self.loss.append(loss)
        self.model.append(model)

    def get_model(self):
        ind = np.argmin(self.loss)
        return self.model[ind]


class FeedBackModel(nn.Module):
    def __init__(self, model_path):
        super(FeedBackModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.linear = nn.Linear(1024, 3)

    def forward(self, ids, mask):
        x = self.model(ids, mask)[0][:, 0, :]
        pred = self.linear(x)
        return pred


class FeedBackDataset(Dataset):
    def __init__(self, data, model_path, is_test=False):
        self.data = data
        self.is_test = is_test
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __getitem__(self, idx):
        text = self.data['discourse_text'].values[idx] + ' ' + \
               self.tokenizer.sep_token * 2 + ' ' + \
               self.data['essay'].values[idx]

        if not self.is_test:
            target_value = self.data[y_cols].values[idx]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN)['input_ids']

        mask = [1] * len(inputs) + [0] * (MAX_LEN - len(inputs))

        mask = torch.tensor(mask, dtype=torch.long)
        if len(inputs) != MAX_LEN:
            inputs = inputs + [self.tokenizer.pad_token_id] * (
                    MAX_LEN - len(inputs))
        ids = torch.tensor(inputs, dtype=torch.long)

        if self.is_test:
            return {
                'ids': ids,
                'mask': mask
            }

        else:
            targets = torch.FloatTensor(target_value)
            return {
                'ids': ids,
                'mask': mask,
                'targets': targets
            }

    def __len__(self):
        return len(self.data)


# reloading the training data
df = pd.read_csv("data/train.csv")

df['essay'] = df['essay_id'].apply(fetch_essay)

new_label = {
    "Ineffective": 0,
    "Adequate": 1,
    "Effective": 2
}

df['discourse_effectiveness'] = df['discourse_effectiveness'].apply(
    lambda x: new_label[x]
)

model_path = 'model_zoo/roberta_large'

y_cols = ['discourse_effectiveness']


# roberta large training
def train_model():
    """

    :return:
    """
    for i, (train_idx, valid_idx) in enumerate(
            StratifiedKFold(n_splits=FOLDS).split(df, y=df['essay_id'])):
        print(f'fold {i + 1}')
        gc.collect()

        train_loader = torch.utils.data.DataLoader(
            FeedBackDataset(
                df.loc[train_idx, :].reset_index(drop=True),
                model_path),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )

        val_loader = torch.utils.data.DataLoader(
            FeedBackDataset(
                df.loc[valid_idx,
                :].reset_index(drop=True),
                model_path),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )

        net = FeedBackModel(model_path)

        net.cuda()

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(net.parameters(), lr=lr)
        param_optimizer = list(net.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in
                                                    param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01}, {
                                            'params': [p for n, p in
                                                       param_optimizer if any(
                                                    nd in n for nd in
                                                    no_decay)],
                                            'weight_decay': 0.0}]
        num_train_optimization_steps = int(
            EPOCHS * len(train_loader) / accumulation_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.05 * num_train_optimization_steps,
            num_training_steps=num_train_optimization_steps
        )

        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(EPOCHS):
            start_time = time.time()
            avg_loss = 0.0
            net.train()
            tbar = tqdm(train_loader)
            loss_list = []
            val_loss_list = []

            for step, data in enumerate(tbar):
                input_ids = data['ids'].cuda()
                input_masks = data['mask'].cuda()
                targets = data['targets'].long().view(-1).cuda()
                with torch.cuda.amp.autocast():
                    pred = net(input_ids, input_masks)
                    loss = loss_fn(pred, targets)
                scaler.scale(loss).backward()
                if step % accumulation_steps == 0 or step == len(tbar) - 1:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                loss_list.append(loss.detach().cpu().item())
                avg_loss = np.round(np.mean(loss_list), 4)
                tbar.set_description(
                    f"Epoch {epoch + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")
            net.eval()
            avg_val_loss = 0.0
            tbar_val = tqdm(val_loader)
            for step, data in enumerate(tbar_val):
                input_ids = data['ids'].cuda()
                input_masks = data['mask'].cuda()
                targets = data['targets'].long().view(-1).cuda()
                pred = net(input_ids, input_masks)
                loss = loss_fn(pred, targets)
                val_loss_list.append(loss.detach().cpu().item())
                avg_val_loss = np.round(np.mean(val_loss_list), 4)
                tbar_val.set_description(
                    f"Epoch {epoch + 1} Loss: {avg_val_loss}")
            with open('roberta_large_epoch_%s_fold_%s.pkl' % (epoch, i),
                      'wb') as f:
                pickle.dump(net.to(torch.device("cuda")), f)
        torch.cuda.empty_cache()

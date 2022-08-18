from roberta_base_config import *


# utils
def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


"""
def get_score(outputs, labels):
    return log_loss(labels, outputs)
"""


def get_score(outputs, labels):
    outputs = F.softmax(torch.tensor(outputs)).numpy()
    return log_loss(labels, outputs)


def get_logger(filename=OUTPUT_DIR + 'train'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()


def seed_everything(seed=CFG.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)

# DataLoading
train = pd.read_csv(INPUT_DIR + 'train_all.csv')
test = pd.read_csv(INPUT_DIR + 'test_all.csv')
print(train.head())
print(train.shape)
print(test.head())
print(test.shape)

# cv splits
skf = StratifiedKFold(
    n_splits=CFG.n_splits,
    shuffle=True,
    random_state=CFG.seed
)

train['fold'] = -1

train['label'] = train['discourse_effectiveness'].map(
    {
        'Ineffective': 0,
        'Adequate': 1,
        'Effective': 2
    }
)

for i, (_, val_) in enumerate(skf.split(train, train['label'])):
    train.loc[val_, 'fold'] = int(i)
train.fold.value_counts()

if CFG.debug:
    print(train.groupby('fold').size())
    train = train.sample(n=1000, random_state=0).reset_index(drop=True)
    print(train.groupby('fold').size())

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(OUTPUT_DIR + 'tokenizer/')
CFG.tokenizer = tokenizer

# dataset
train['text'] = train['discourse_text'] + '[SEP]' + train['essay_text']


class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = CFG.max_len
        self.text = df['text'].values
        self.tokenizer = CFG.tokenizer
        self.targets = df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len
        )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': self.targets[index]
        }


collate_fn = DataCollatorWithPadding(tokenizer=CFG.tokenizer)


# model
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, CFG.target_size)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids,
                         attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs


#  helper function
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "%s (remain %s)" % (as_minutes(s), as_minutes(rs))


def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles
        )
    return scheduler


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0

    running_loss = 0

    start = end = time.time()

    for step, data in enumerate(dataloader):
        ids = data['input_ids'].to(device, dtype=torch.long)

        mask = data['attention_mask'].to(device, dtype=torch.long)

        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)

        # accumulate
        loss = loss / CFG.n_accumulate

        loss.backward()

        if (step + 1) % CFG.n_accumulate == 0:
            optimizer.step()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)

        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        end = time.time()

        if step % CFG.print_freq == 0 or step == (len(dataloader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
            .format(
                epoch + 1,
                step,
                len(dataloader),
                remain=time_since(start,
                                  float(step + 1) / len(dataloader))))

    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0

    running_loss = 0

    start = end = time.time()

    pred = []

    for step, data in enumerate(dataloader):
        ids = data['input_ids'].to(device, dtype=torch.long)

        mask = data['attention_mask'].to(device, dtype=torch.long)

        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)

        pred.append(outputs.to('cpu').numpy())

        running_loss += (loss.item() * batch_size)

        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        end = time.time()

        if step % CFG.print_freq == 0 or step == (len(dataloader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
            .format(
                step,
                len(dataloader),
                remain=time_since(
                    start,
                    float(step + 1) / len(dataloader)
                )
            ))

    pred = np.concatenate(pred)

    return epoch_loss, pred


def train_loop(fold):
    # wandb.watch(model, log_freq=100)

    LOGGER.info(f'-------------fold:{fold} training-------------')

    train_data = train[train.fold != fold].reset_index(drop=True)

    valid_data = train[train.fold == fold].reset_index(drop=True)

    valid_labels = valid_data.label.values

    train_dataset = FeedBackDataset(
        train_data,
        CFG.tokenizer,
        CFG.max_len
    )

    valid_dataset = FeedBackDataset(
        valid_data,
        CFG.tokenizer,
        CFG.max_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = FeedBackModel(CFG.model)
    torch.save(model.config, OUTPUT_DIR + 'config.pth')
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weigth_decay
    )

    num_train_steps = int(len(train_data) / CFG.batch_size * CFG.epochs)

    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # loop
    best_score = 100

    for epoch in range(CFG.epochs):
        start_time = time.time()

        train_epoch_loss = train_one_epoch(
            model,
            optimizer,
            scheduler,
            train_loader,
            device,
            epoch
        )

        valid_epoch_loss, pred = valid_one_epoch(
            model,
            valid_loader,
            device,
            epoch
        )

        score = get_score(
            pred,
            valid_labels
        )

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {train_epoch_loss:.4f} avg_val_loss: {valid_epoch_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}')
        if CFG.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch + 1,
                       f"[fold{fold}] avg_train_loss": train_epoch_loss,
                       f"[fold{fold}] avg_val_loss": valid_epoch_loss,
                       f"[fold{fold}] score": score})

        if score < best_score:
            best_score = score
            LOGGER.info(
                f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': pred},
                       OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(
        OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device('cpu'))['predictions']
    valid_data['pred_0'] = predictions[:, 0]
    valid_data['pred_1'] = predictions[:, 1]
    valid_data['pred_2'] = predictions[:, 2]
    torch.cuda.empty_cache()
    gc.collect()
    return valid_data

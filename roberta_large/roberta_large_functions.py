from roberta_large_CFG import *


# ====================================================
# Utils
# ====================================================

def fetchEssay(essay_id: str):
    essay_path = os.path.join('../input/feedback-prize-effectiveness/train/', essay_id + '.txt')
    essay_text = open(essay_path, 'r').read()
    return essay_text


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_essay(essay_id, is_train=True):
    parent_path = INPUT_DIR + 'train' if is_train else INPUT_DIR + 'test'
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def get_score(y_true, y_pred):
    y_pred = softmax(y_pred)
    score = log_loss(y_true, y_pred)
    return round(score, 5)


test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
test['essay_text'] = test['essay_id'].apply(lambda x: get_essay(x, is_train=False))

# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.path + 'tokenizer')
tokenizer = CFG.tokenizer


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start: error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start: error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


test['discourse_text'] = test['discourse_text'].apply(lambda x: resolve_encodings_and_normalize(x))
test['essay_text'] = test['essay_text'].apply(lambda x: resolve_encodings_and_normalize(x))

SEP = tokenizer.sep_token
test['text'] = test['discourse_type'] + ' ' + test['discourse_text'] + SEP + test['essay_text']
test['label'] = np.nan
print(test.head())


#
class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text = df['text'].values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = self.cfg.tokenizer.encode_plus(
            self.text[item],
            truncation=True,
            add_special_tokens=True,
            max_length=self.cfg.max_len
        )
        samples = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
        }
        if 'token_type_ids' in inputs:
            samples['token_type_ids'] = inputs['token_type_ids']
        return samples


class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        # self.args = args

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in
                                   output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in
                                   output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output


collate_fn = Collate(CFG.tokenizer, isTrain=False)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        _, max_pooling_embeddings = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat((mean_pooling_embeddings, max_pooling_embeddings), 1)
        return mean_max_embeddings


class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, all_hidden_states):
        # forward
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out


# weighted_pooling
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # gradient checkpointing
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()
            print(f"Gradient Checkpointing: {self.model.is_gradient_checkpointing}")

        # self.pooler = MeanPooling()
        self.bilstm = nn.LSTM(
            self.config.hidden_size,
            self.config.hidden_size // 2,
            num_layers=2,
            dropout=self.config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self.output = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.cfg.target_size)
            # nn.Linear(256, self.cfg.target_size)
        )

    def loss(self, outputs, targets):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, targets)
        return loss

    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        # print(outputs)
        # print(targets)
        mll = log_loss(
            targets.cpu().detach().numpy(),
            softmax(outputs.cpu().detach().numpy()),
            labels=[0, 1, 2],
        )
        return mll

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.model(ids, mask, token_type_ids)
        else:
            transformer_out = self.model(ids, mask)

        # LSTM/GRU header
        #         all_hidden_states = torch.stack(transformer_out[1])
        #         sequence_output = self.pooler(all_hidden_states)

        # simple CLS
        sequence_output = transformer_out[0][:, 0, :]

        # Main task
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        if targets is not None:
            metric = self.monitor_metrics(logits, targets)
            return logits, metric

        return logits, 0.


seed_everything(SEED)


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
        text = self.data['discourse_text'].values[idx] + ' ' + self.tokenizer.sep_token * 2 + ' ' + \
               self.data['essay'].values[idx]
        if not self.is_test:
            target_value = self.data[y_cols].values[idx]

        inputs = self.tokenizer.encode_plus(text, None, truncation=True, add_special_tokens=True, max_length=MAX_LEN)[
            'input_ids']
        mask = [1] * len(inputs) + [0] * (MAX_LEN - len(inputs))
        mask = torch.tensor(mask, dtype=torch.long)
        if len(inputs) != MAX_LEN: inputs = inputs + [self.tokenizer.pad_token_id] * (MAX_LEN - len(inputs))
        ids = torch.tensor(inputs, dtype=torch.long)

        if self.is_test:
            return {'ids': ids, 'mask': mask, }

        else:
            targets = torch.FloatTensor(target_value)
            return {'ids': ids, 'mask': mask, 'targets': targets}

    def __len__(self):
        return len(self.data)


def train_model():
    for i, (train_idx, valid_idx) in enumerate(StratifiedKFold(n_splits=FOLDS).split(df, y=df['essay_id'])):
        print(f'fold {i + 1}')
        gc.collect()

        train_loader = torch.utils.data.DataLoader(
            FeedBackDataset(df.loc[train_idx, :].reset_index(drop=True), model_path), batch_size=BATCH_SIZE,
            shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(
            FeedBackDataset(df.loc[valid_idx, :].reset_index(drop=True), model_path), batch_size=BATCH_SIZE,
            shuffle=False, num_workers=2)
        net = FeedBackModel(model_path)
        net.cuda()

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(net.parameters(), lr=lr)
        param_optimizer = list(net.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        num_train_optimization_steps = int(EPOCHS * len(train_loader) / accumulation_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                    num_training_steps=num_train_optimization_steps)
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
                tbar.set_description(f"Epoch {epoch + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")
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
                tbar_val.set_description(f"Epoch {epoch + 1} Loss: {avg_val_loss}")
            with open('roberta_large_epoch_%s_fold_%s.pkl' % (epoch, i), 'wb') as f:
                pkl.dump(net.to(torch.device("cpu")), f)
        torch.cuda.empty_cache()


df = pd.read_csv("../data/train.csv")
df['essay'] = df['essay_id'].apply(fetchEssay)
new_label = {"Ineffective": 0, "Adequate": 1, "Effective": 2}
df['discourse_effectiveness']  = df['discourse_effectiveness'].apply(lambda x: new_label[x] )

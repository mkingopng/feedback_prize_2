from he_deberta_base_CFG import *


def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_freezed_parameters(module):
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
    return freezed_parameters


def get_essay(essay_id, is_train=True):
    parent_path = INPUT_DIR + 'train' if is_train else INPUT_DIR + 'test'
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text


# preprocessing
# testing data
test = pd.read_csv(INPUT_DIR + 'test.csv')

test['essay_text'] = test['essay_id'].apply(lambda x: get_essay(x, is_train=False))

if CFG.fast:
    tokenizer = AutoTokenizer.from_pretrained(CFG.model, use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)

CFG.tokenizer = tokenizer


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start: error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start: error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    text = (text.encode("raw_unicode_escape").decode("utf-8", errors="replace_decoding_with_cp1252").encode("cp1252",
                                                                                                            errors="replace_encoding_with_utf8").decode(
        "utf-8", errors="replace_decoding_with_cp1252"))
    text = unidecode(text)
    return text


test['discourse_text'] = test['discourse_text'].apply(lambda x: resolve_encodings_and_normalize(x))

test['essay_text'] = test['essay_text'].apply(lambda x: resolve_encodings_and_normalize(x))

test['text'] = test['discourse_type'] + ' ' + test['discourse_text'] + '[SEP]' + test['essay_text']

test['label'] = np.nan


# dataset and dynamic padding
class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text = df['text'].values

    def __len__(self): return len(self.text)

    def __getitem__(self, item):
        inputs = self.cfg.tokenizer.encode_plus(self.text[item],
                                                truncation=True,
                                                add_special_tokens=True,
                                                max_length=self.cfg.max_len
                                                )

        samples = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], }
        if 'token_type_ids' in inputs:
            samples['token_type_ids'] = inputs['token_type_ids']
        return samples


class Collate:
    def __init__(self, tokenizer, is_train=True):
        self.isTrain = is_train
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain: output["target"] = [sample["target"] for sample in batch]
        batch_max = max([len(ids) for ids in output["input_ids"]])
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in
                                   output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in
                                   output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)

        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)

        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output


# the model
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  #
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for data in tk0:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        with torch.no_grad():
            y_preds = model(ids, mask)
        y_preds = softmax(y_preds.to('cpu').numpy())
        preds.append(y_preds)
    predictions = np.concatenate(preds)
    return predictions


class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if CFG.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if CFG.freezing:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:2])
            CFG.after_freezed_parameters = filter(lambda parameter: parameter.requires_grad, self.model.parameters())
        self.config = AutoConfig.from_pretrained(model_name)
        self.drop = nn.Dropout(p=CFG.dropout)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, CFG.target_size)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs


# deberta-base inference
testDataset = TestDataset(CFG, test)
test_loader = DataLoader(
    testDataset,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    batch_size=CFG.batch_size,
    num_workers=CFG.num_workers,
    collate_fn=Collate(CFG.tokenizer, is_train=False)
)

deberta_predictions = []
for i in CFG.trn_fold:
    model = FeedBackModel(CFG.model)
    model.load_state_dict(torch.load(
        'model_zoo/dbv3basemodels202279/models-deberta-v3-base-deberta-v3-base_fold' + str(i) + '_best.pth'))
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

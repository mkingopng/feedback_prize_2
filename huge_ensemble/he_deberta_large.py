from he_deberta_large_CFG import *
from he_helper_functions import *


# data loading
N_ROW = 10

props_param = "color:white; font-weight:bold; background-color:green;"

test_path = COMP_DIR + "test.csv"
submission_path = COMP_DIR + "sample_submission.csv"
test_origin = pd.read_csv(test_path)
submission_origin = pd.read_csv(submission_path)
data_path = "../data/train.csv"
cols_list = ['essay_id', 'discourse_text']
idxs_list = [49, 80, 945, 947, 1870]
temp = pd.read_csv(data_path, usecols=cols_list).loc[idxs_list, :]

temp['discourse_text_UPD'] = temp['discourse_text'].apply(resolve_encodings_and_normalize)
temp['essay_text'] = temp['essay_id'].transform(fetch_essay, txt_dir='train')
temp['essay_text_UPD'] = temp['essay_text'].apply(resolve_encodings_and_normalize)

for n, row in enumerate(temp.iterrows()):
    indx, data = row
    disc_text = data.discourse_text
    disc_text_upd = data.discourse_text_UPD
    print(f'\nN{n} === index: {indx} ===')
    print(f'\n>>> origin text:')
    print(repr(disc_text))
    print(f'\n>>> updated text:')
    print(repr(disc_text_upd))


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text = df['text'].values

    def __len__(self): return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        inputs = prepare_input(self.cfg, text)
        return inputs


# the model
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
        self.bilstm = nn.LSTM(
            self.config.hidden_size,
            self.config.hidden_size // 2,
            num_layers=2,
            dropout=self.config.hidden_dropout_prob,
            batch_first=True, bidirectional=True
        )
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Sequential(nn.Linear(self.config.hidden_size, 3))

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

    def forward(self, inputs):
        sequence_output = self.model(**inputs)[0][:, 0, :]
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        return logits


CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path + 'tokenizer')

df = test_origin.copy()
SEP = CFG.tokenizer.sep_token
df['discourse_text'] = df['discourse_text'].apply(resolve_encodings_and_normalize)
df['essay_text'] = df['essay_id'].transform(fetch_essay, txt_dir='test')
df['essay_text'] = df['essay_text'].apply(resolve_encodings_and_normalize)
df['text'] = df['discourse_type'] + ' ' + df['discourse_text'] + SEP + df['essay_text']

test_dataset = TestDataset(CFG, df)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    drop_last=False
)


# deberta large inference
deberta_large_predictions = []
for fold in range(CFG.n_fold):
    model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
    state = torch.load(
        CFG.path + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device('cpu')
    )
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, DEVICE)
    deberta_large_predictions.append(prediction)
    del model, state, prediction; gc.collect()
    torch.cuda.empty_cache()

deb_large_adequate = []
deb_large_effective = []
deb_large_ineffective = []

for x in deberta_large_predictions:
    deb_large_ineffective.append(x[:, 0])
    deb_large_adequate.append(x[:, 1])
    deb_large_effective.append(x[:, 2])

deb_large_ineffective = pd.DataFrame(deb_large_ineffective).T
deb_large_adequate = pd.DataFrame(deb_large_adequate).T
deb_large_effective = pd.DataFrame(deb_large_effective).T


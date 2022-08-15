from he_roberta_large_CFG import *
from he_helper_functions import *


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.essay = df['essay'].values
        self.discourse = df['discourse'].values

    def __len__(self): return len(self.discourse)

    def __getitem__(self, item):
        discourse = self.discourse[item]
        essay = self.essay[item]
        inputs = prepare_input(self.cfg, discourse, essay)
        return inputs


class FeedBackModel(nn.Module):
    def __init__(self, model_path):
        super(FeedBackModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.linear = nn.Linear(1024, 3)

    def forward(self, inputs):
        last_hidden_states = self.model(**inputs)[0][:, 0, :]
        outputs = self.linear(last_hidden_states)
        return outputs


CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path)

df = test_origin.copy()

txt_sep = " "
df['discourse'] = df['discourse_type'].str.strip() + txt_sep + df['discourse_text'].str.strip()
df['essay'] = df['essay_id'].transform(fetch_essay, txt_dir='test').str.strip()

test_dataset = TestDataset(CFG, df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    drop_last=False
)

# roberta large inference
gc.collect()
roberta_predicts = []
for model_path in os.listdir('../input/feedback-roberta-models'):
    if 'data_1' in model_path:
        model = pkl.load(open('../input/feedback-roberta-models/' + model_path, 'rb'))
        prediction = inference_fn(test_loader, model, DEVICE)
        roberta_predicts.append(prediction)
        del model, prediction
        torch.cuda.empty_cache()
        gc.collect()
gc.collect()

rob_adequate = []
rob_effective = []
rob_ineffective = []

for x in roberta_predicts:
    rob_ineffective.append(x[:, 0])
    rob_adequate.append(x[:, 1])
    rob_effective.append(x[:, 2])

rob_ineffective = pd.DataFrame(rob_ineffective).T
rob_adequate = pd.DataFrame(rob_adequate).T
rob_effective = pd.DataFrame(rob_effective).T

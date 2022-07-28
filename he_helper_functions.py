from he_imports import *


# helper functions
def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start: error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start: error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)

codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    text = (text.encode("raw_unicode_escape").decode("utf-8", errors="replace_decoding_with_cp1252").encode("cp1252", errors="replace_encoding_with_utf8").decode("utf-8", errors="replace_decoding_with_cp1252"))
    text = unidecode(text)
    return text


def fetch_essay(essay_id: str, txt_dir: str):
    essay_path = os.path.join(COMP_DIR + txt_dir, essay_id + '.txt')
    essay_text = open(essay_path, 'r').read()
    return essay_text


def prepare_input(cfg, text, text_2=None):
    inputs = cfg.tokenizer(
        text,
        text_2,
        padding="max_length",
        add_special_tokens=True,
        max_length=cfg.max_len,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            output = model(inputs)
        preds.append(F.softmax(output).to('cpu').numpy())
    return np.concatenate(preds)


def show_gradient(df, n_row=None):
    if not n_row:
        n_row = 5
    return df.head(n_row).assign(all_mean=lambda x: x.mean(axis=1)).style.background_gradient(cmap=cm, axis=1)

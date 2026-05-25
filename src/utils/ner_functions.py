import re
import torch
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader
_THIS_FILE = Path(__file__).resolve()


def _resolve_project_root() -> Path:
    for candidate in _THIS_FILE.parents:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    return _THIS_FILE.parents[2]


PROJECT_ROOT = _resolve_project_root()
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from ..Models.BiLSTM_CRF import BiLSTM_CRF, STOP_TAG, START_TAG
except ImportError:
    from Models.BiLSTM_CRF import BiLSTM_CRF, STOP_TAG, START_TAG

try:
    from .utils import get_chunks, prepare_char_inputs, lower_case
except ImportError:
    from utils import get_chunks, prepare_char_inputs, lower_case


class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = torch.LongTensor(item['words'])
        tags = torch.LongTensor(item['tags'])
        chars = [torch.LongTensor(c) for c in item['chars']]
        return words, chars, tags


def custom_collate_fn(batch):
    return batch[0]


def train_loop(model, train_data, optimizer, grad_clip, device, writer, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for words, chars, tags in train_data:
        model.zero_grad()

        words = words.to(device)
        tags = tags.to(device)
        chars_mask, chars_len, d = prepare_char_inputs(
            chars, device, model.char_mode)

        loss = model.neg_log_likelihood(words, tags, chars_mask, chars_len, d)
        loss.backward()

        _, out = model(words, chars_mask, chars_len, d)
        all_preds.append(out)
        all_targets.append(tags.tolist())

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)
    f1 = compute_f1(all_preds, all_targets, model.tag_to_ix)

    writer.add_scalar("Loss/Train", avg_loss, epoch)
    writer.add_scalar("F1/Train", f1, epoch)
    return avg_loss, f1


def val_loop(model, val_data, device, writer, epoch):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for words, chars, tags in val_data:
            words = words.to(device)
            tags = tags.to(device)
            chars_mask, chars_len, d = prepare_char_inputs(
                chars, device, model.char_mode)

            loss = model.neg_log_likelihood(
                words, tags, chars_mask, chars_len, d)
            total_loss += loss.item()

            _, out = model(words, chars_mask, chars_len, d)
            all_preds.append(out)
            all_targets.append(tags.tolist())

    avg_loss = total_loss / len(val_data)
    f1 = compute_f1(all_preds, all_targets, model.tag_to_ix)

    writer.add_scalar("Loss/Val", avg_loss, epoch)
    writer.add_scalar("F1/Val", f1, epoch)
    return avg_loss, f1


def compute_f1(preds, targets, tag_to_ix):
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for p, t in zip(preds, targets):
        lab_chunks = set(get_chunks(t, tag_to_ix))
        lab_pred_chunks = set(get_chunks(p, tag_to_ix))
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if total_preds > 0 else 0
    r = correct_preds / total_correct if total_correct > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def adjust_learning_rate(optimizer, lr):
    """
    Updates the learning rate for all parameter groups in the optimizer.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Learning rate adjusted to: {lr:.6f}")


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\\d', '0', s)


def load_sentences(path, zeros, ):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for i, line in enumerate(open(path, 'r', encoding='utf-8')):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def iob2(tags):
    """
    Check that tags have a valid BIO format.
    Tags in BIO1 format are converted to BIO2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    the function is used to convert
    BIO -> BIOES tagging
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to BIO2
    Only BIO1 and BIO2 schemes are accepted for input data.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the BIO format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in BIO format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'BIOES':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Wrong tagging scheme!')


def create_dic(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def word_mapping(sentences, lower, debug=False):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dic(words)
    dico['<UNK>'] = 10000000  # UNK tag for unknown words
    word_to_id, id_to_word = create_mapping(dico)
    if debug:
        print("Found %i unique words (%i in total)" % (
            len(dico), sum(len(x) for x in words)
        ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences, debug=False):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dic(chars)
    char_to_id, id_to_char = create_mapping(dico)
    if debug:
        print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences, debug=False):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dic(tags)
    dico[START_TAG] = -1
    dico[STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    if debug:
        print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[lower_case(w, lower) if lower_case(w, lower) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags': tags,
        })
    return data


def load_conll_data(train_path: str, dev_path: str, test_path: str, params: dict, debug: bool = False):
    """
    Complete pipeline for training, validation and test data: Load -> Update Tags -> Create Mappings -> Prepare Dataset
    """
    train_sents = load_sentences(train_path, params['zeros'])
    dev_sents = load_sentences(dev_path, params['zeros'])
    test_sents = load_sentences(test_path, params['zeros'])

    if debug:
        print('train sentences: %d' % len(train_sents))
        print('test sentences: %d' % len(test_sents))
        print('dev sentences: %d' % len(dev_sents))

        print("train_sentences[0]: ", train_sents[0])

    update_tag_scheme(train_sents, params['tag_scheme'])
    update_tag_scheme(dev_sents, params['tag_scheme'])
    update_tag_scheme(test_sents, params['tag_scheme'])

    dico_words, word_to_id, id_to_word = word_mapping(
        train_sents, params['lower'], debug=debug)
    dico_chars, char_to_id, id_to_char = char_mapping(train_sents, debug=debug)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sents, debug=debug)

    train_data = prepare_dataset(
        train_sents, word_to_id, char_to_id, tag_to_id, params['lower'])
    dev_data = prepare_dataset(
        dev_sents, word_to_id, char_to_id, tag_to_id, params['lower'])
    test_data = prepare_dataset(
        test_sents, word_to_id, char_to_id, tag_to_id, params['lower'])

    train_dataset = NERDataset(train_data)
    dev_dataset = NERDataset(dev_data)
    test_dataset = NERDataset(test_data)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=1,
                            shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, dev_loader, test_loader, (word_to_id, char_to_id, tag_to_id)


def build_model(params: dict, debug: bool = False):
    _, _, _, (word_to_id, char_to_id, tag_to_id) = load_conll_data(
        params["train"], params["dev"], params["test"], params, debug=debug
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_CRF(
        vocab_size=len(word_to_id),
        tag_to_ix=tag_to_id,
        embedding_dim=params["word_dim"],
        hidden_dim=params["word_lstm_dim"],
        char_to_ix=char_to_id,
        char_lstm_dim=params["char_lstm_dim"],
        dropout=params["dropout"],
        char_mode=params["char_mode"],
        use_gpu=torch.cuda.is_available(),
        device=device,
    ).to(device)

    reload_path = params.get("reload") if params else None
    if reload_path:
        model_path = Path(reload_path).expanduser()
    else:
        model_path = PROJECT_ROOT / "models" / "ner_models" / "best_ner_model.pth"

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"NER checkpoint not found at '{model_path}'. "
            "Set params['reload'] to a valid checkpoint path or place the file at "
            f"'{PROJECT_ROOT / 'models' / 'ner_models' / 'best_ner_model.pth'}'."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, word_to_id, char_to_id, tag_to_id, params, device


def load_pretrained_embeddings(path, word_to_id, word_dim):
    """ Load pretrained embeddings from a file and create an embedding matrix for the words in the dataset."""
    all_word_embeds = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(x) for x in s[1:]])

    scale = np.sqrt(0.06)
    word_embeds = torch.empty(len(word_to_id), word_dim)
    torch.nn.init.uniform_(word_embeds, -scale, scale)

    count = 0
    for w, i in word_to_id.items():
        if w in all_word_embeds:
            word_embeds[i] = torch.FloatTensor(all_word_embeds[w])
            count += 1
        elif w.lower() in all_word_embeds:
            word_embeds[i] = torch.FloatTensor(all_word_embeds[w.lower()])
            count += 1

    print(f'Loaded {count} pretrained embeddings out of {len(word_to_id)}.')
    return word_embeds


def save_mappings(mapping_file, word_to_id, tag_to_id, char_to_id, parameters, word_embeds):
    """ Save mappings and parameters to a file."""
    with open(mapping_file, 'wb') as f:
        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'char_to_id': char_to_id,
            'parameters': parameters,
            'word_embeds': word_embeds
        }
        pickle.dump(mappings, f)
    return mappings


def flatten_tags(sentences):
    return [word[-1] for sent in sentences for word in sent]


def split_metrics(sentences):
    tokens = [word[0] for sent in sentences for word in sent]
    tags = flatten_tags(sentences)
    sent_lengths = [len(sent) for sent in sentences]

    entity_tags = [tag for tag in tags if tag != "O"]
    entity_types = [
        tag.split("-", 1)[1] if "-" in tag else tag for tag in entity_tags]

    return {
        "sentences": len(sentences),
        "tokens": len(tokens),
        "avg_sentence_len": sum(sent_lengths) / len(sent_lengths),
        "entity_types": len(set(entity_types)),
        "pct_O": 100.0 * tags.count("O") / len(tags),
        "pct_entity": 100.0 * len(entity_tags) / len(tags),
    }


def plot_label_distribution(label_dist_df: pd.DataFrame, summary_df: pd.DataFrame):
    labels_for_plot = [
        label for label in label_dist_df.index if label != "O"][:20]
    if labels_for_plot:
        ax = label_dist_df.loc[labels_for_plot].plot(
            kind="bar", figsize=(14, 5))
        ax.set_title("Top 20 non-O labels by count")
        ax.set_xlabel("Label")
        ax.set_ylabel("Token count")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    ax = summary_df[["pct_O", "pct_entity"]].plot(kind="bar", figsize=(8, 4))
    ax.set_title("O vs Entity token ratio")
    ax.set_xlabel("Split")
    ax.set_ylabel("Percentage")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def predict_ner(text, model, word_to_id, char_to_id, tag_to_id, params, device):
    tokens = text.split()
    word_tokens = [lower_case(zero_digits(token), params["lower"])
                   for token in tokens]
    char_tokens = [zero_digits(token) for token in tokens]

    words = [word_to_id.get(token, word_to_id["<UNK>"])
             for token in word_tokens]
    chars = [[char_to_id[c] for c in token if c in char_to_id]
             for token in char_tokens]

    sentence_tensor = torch.LongTensor(words).to(device)
    chars_mask, chars_len, d = prepare_char_inputs(
        chars, device, params["char_mode"])

    with torch.no_grad():
        _, tag_ids = model(sentence_tensor, chars_mask, chars_len, d)

    id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}
    return list(zip(tokens, [id_to_tag[tag_id] for tag_id in tag_ids]))


def normalize_ner_output(ner_data):
    """
    Normalizes BIOES tagged NER output into a readable string.
    Input: list of (token, tag) tuples
    """
    if isinstance(ner_data, str):
        try:
            ner_data = ast.literal_eval(ner_data)
        except Exception:
            ner_data

    tag_map = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        "MISC": "MISCELLANEOUS"
    }

    entities = []
    current_entity_tokens = []
    current_type = None

    for token, tag in ner_data:
        if tag.startswith("S-"):
            entity_type = tag.split("-")[1]
            readable_type = tag_map.get(entity_type, entity_type)
            entities.append(f"{token} ({readable_type})")

        elif tag.startswith("B-"):
            current_type = tag.split("-")[1]
            current_entity_tokens = [token]

        elif tag.startswith("I-"):
            current_entity_tokens.append(token)

        elif tag.startswith("E-"):
            current_entity_tokens.append(token)
            full_name = " ".join(current_entity_tokens)
            readable_type = tag_map.get(current_type, current_type)
            entities.append(f"{full_name} ({readable_type})")
            current_entity_tokens = []
            current_type = None

        else:
            continue
    return ", ".join(entities)

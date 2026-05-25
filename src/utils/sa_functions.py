import torch
from typing import List, Any
import sys
import os
from pathlib import Path
from gensim.models.keyedvectors import load_word2vec_format
try:
    from .utils import download_and_unzip_dataset_from_url, tokenize_tweet
except ImportError:
    from utils import download_and_unzip_dataset_from_url, tokenize_tweet

MODEL_NAME = "ConvBiLSTM_v2.0.0_12-04-2026"

# ----- WORD2VEC PRETRAINED MODEL -----
WORD2VEC = "word2vec"
word2vec_url = "https://drive.google.com/uc?id=1zQRH1zYBHJ_vU_uMkKvvvwQiZwP5N7wW"
word2vec_zip = "word2vec.zip"
known_word2vec_gz_file = "GoogleNews-vectors-negative300"
data_dir = "../data"


def word2idx(embedding_model: Any, tweet: List[str]) -> torch.Tensor:
    """
    Converts a tweet to a list of word indices based on an embedding model.

    This function iterates through each word in the tweet and retrieves its corresponding index
    from the embedding model's vocabulary. If a word is not present in the model's vocabulary,
    it is skipped.

    Args:
        embedding_model (Any): The embedding model with a 'key_to_index' attribute, which maps words to their indices.
        tweet (List[str]): A list of words representing the tweet.

    Returns:
        torch.Tensor: A tensor of word indices corresponding to the words in the tweet.
    """
    return torch.tensor([embedding_model.key_to_index[t] for t in tweet if t in embedding_model.key_to_index])


def load_sa_model(name: str | os.PathLike[str], debug=False):
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent.parent
    name_path = Path(name).expanduser()

    candidate_paths = []
    if name_path.is_absolute() or name_path.parent != Path("."):
        candidate_paths.append(name_path)
        if name_path.suffix != ".pt":
            candidate_paths.append(Path(f"{name_path}.pt"))
    else:
        candidate_paths.append(project_root / "models" /
                               "sa_models" / f"{name_path.name}.pt")
        if name_path.suffix == ".pt":
            candidate_paths.append(
                project_root / "models" / "sa_models" / name_path.name)
        else:
            candidate_paths.append(
                project_root / "models" / "sa_models" / name_path.name)
            candidate_paths.append(project_root / f"{name_path.name}.pt")
            candidate_paths.append(project_root / name_path.name)

    model_path = next(
        (path for path in candidate_paths if path.exists()), None)
    if model_path is None:
        searched_paths = "\n".join(f"- {path}" for path in candidate_paths)
        raise FileNotFoundError(
            "Sentiment analysis model checkpoint not found. Looked in:\n"
            f"{searched_paths}"
        )

    with open(model_path, 'rb') as f:
        model = torch.jit.load(f)

    if debug:
        print("[+] Loading embeddings...")

    data_directory = str(project_root / "data")
    word2vec_files: list[str] = download_and_unzip_dataset_from_url(
        word2vec_url,
        data_directory,
        word2vec_zip,
        WORD2VEC,
        force_download_and_unzip=False,
        use_gdown=True,
        delete_MACOSX=True,
        debug=debug,
    )
    word2vec_model = None

    for file in word2vec_files:
        if known_word2vec_gz_file in file:
            word2vec_model = file

    if word2vec_model is None:
        raise ValueError("Path to model could not be found.")

    w2v_model = load_word2vec_format(word2vec_model, binary=True)
    return model, w2v_model


def predict_sa(text, model, w2v_model, device, debug=False):
    if debug:
        print("[+] Tokenizing text...")
    tokenized = tokenize_tweet(text)

    vectors = [w2v_model[t] for t in tokenized if t in w2v_model]
    if not vectors:
        vectors = [torch.zeros(w2v_model.vector_size)]
    else:
        vectors = [torch.tensor(v) for v in vectors]

    embedded_tensor = torch.stack(vectors).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()

    if debug:
        print("[+] Predicting sentiment...")
    with torch.no_grad():
        logits = model(embedded_tensor)

    pred = torch.softmax(logits, dim=1).squeeze()
    if debug:
        print(f"[?] Probabilities: {list(pred.detach().cpu().numpy())}")

    out = torch.argmax(pred, dim=0)
    labels = {
        0: "negative",
        1: "positive"
    }

    return labels[out.item()]


if __name__ == "__main__":
    print("[+] Loading model...")
    model, w2v_model = load_sa_model(MODEL_NAME, debug=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_sentence = ' '.join(sys.argv[1:])
    print(f"[+] Detected Sentence: {input_sentence}")

    sentiment = predict_sa(input_sentence, model,
                           w2v_model, device, debug=True)
    print(f"[+] Sentiment detected: {sentiment}")

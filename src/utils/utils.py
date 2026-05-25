import os
import re
from zipfile import ZipFile
from zipfile import is_zipfile
import gdown
import shutil
import torch
from pathlib import Path


user_pattern = re.compile(r"@[\w\_\.\-]*")
url_pattern = re.compile(r"http[s]*:\/\/[\w\.\/\-\+\%\d\:]*")
hashtag_pattern = re.compile(r"#[\w\-!?\_]*")
space_pattern = re.compile(r"\s+")


def download_and_unzip_dataset_from_url(url: str, data_dir: str, zip_name: str, final_dir_name: str, force_download_and_unzip: bool = False, use_gdown: bool = False, delete_MACOSX: bool = False, debug: bool = False) -> list[str]:
    """
    Downloads a zip file to the directory `'data_dir' + 'zip_name'` from the url 
    and unzips it in `'data_dir' + 'final_dir_name'`.

    Args:
        url (str): Url to dataset zip download.
        data_dir (str): Data directory.
        zip_name (str): Target zip name for downloaded zip.
        final_dir_name (str): Directory to unzip the dataset in.
        force_download_and_unzip (bool, optional): Forces the download and unzip. Defaults to False.

    Raises:
        Exception: Dataset could not be downloaded and shows http status code.

    Returns:
        list[str]: path to files unzipped.
    """
    import urllib.request

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    zip_filename = zip_name if zip_name.endswith(".zip") else f"{zip_name}.zip"
    zip_path = data_path / zip_filename
    final_path = data_path / final_dir_name

    def _list_extracted_files() -> list[str]:
        if not final_path.exists():
            return []

        files: list[str] = []
        for extracted_path in final_path.rglob("*"):
            if extracted_path.is_file():
                files.append(str(extracted_path))
        return files

    zip_exists = zip_path.exists() and is_zipfile(zip_path)
    extracted_files = _list_extracted_files()
    extracted_exists = len(extracted_files) > 0

    if not zip_exists or force_download_and_unzip:
        if debug:
            print(f"Downloading dataset {zip_name}...")
        if not use_gdown:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    with open(zip_path, "wb") as out_file:
                        out_file.write(response.read())
                        if debug:
                            print("Dataset downloaded correctly.")
                else:
                    raise Exception(
                        f"Dataset could not be downloaded. Status: {response.status}")
        else:
            downloaded_path = gdown.download(url, str(zip_path), quiet=False)
            if downloaded_path is None or not zip_path.exists():
                raise Exception("Dataset could not be downloaded with gdown.")
    else:
        if debug:
            print(f"Dataset {zip_name} already downloaded, skipping...")

    files: list[str] = []
    if not extracted_exists or force_download_and_unzip:
        if debug:
            print(f"Unzipping dataset {zip_name}...")
        if final_path.exists():
            shutil.rmtree(final_path)
        final_path.mkdir(parents=True, exist_ok=True)

        with ZipFile(zip_path, "r") as zf:
            zf.extractall(final_path)
        if debug:
            print("Dataset unzipped.")
        files = _list_extracted_files()
    else:
        if debug:
            print(f"Dataset {zip_name} already unzipped, skipping...")
        files = extracted_files

    if delete_MACOSX:
        for root, dirs, files_walk in os.walk(final_path):
            for dir in dirs:
                if "MACOSX" in dir:
                    shutil.rmtree(os.path.join(root, dir))
            for file in files_walk:
                if "MACOSX" in file:
                    os.remove(os.path.join(root, file))
        files = _list_extracted_files()

    return [str(Path(file)) for file in files]


def tokenize_tweet(tweet: str) -> list[str]:
    """
    Tokenizes a given tweet by splitting the text into words, and doing any cleaning, replacing or normalization deemed useful

    Args:
        tweet (str): The tweet text to be tokenized.

    Returns:
        list[str]: A list of strings, representing the tokenized components of the tweet.
    """
    tweet = str(tweet)

    tweet = re.sub(user_pattern, "user", tweet)
    tweet = re.sub(url_pattern, "url", tweet)
    tweet = re.sub(hashtag_pattern, "hashtag", tweet)

    tweet = tweet.strip()
    tweet = re.sub(space_pattern, " ", tweet)

    return tweet.split(" ")


def log_sum_exp(vec):
    """
    This function calculates the score necessary for the forward algorithm of the BiLSTM-CRF Model
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    """
    This function returns the max index in a vector
    """
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def to_scalar(var):
    """
    Function to convert pytorch tensor to a scalar
    """
    return var.view(-1).data.tolist()[0]


def get_chunks(seq, tags):
    """
    Given a sequence of tag IDs, group entities and their positions.
    Supports BIO and BIOES schemes.

    Args:
        seq: List of tag IDs (e.g., [4, 5, 0, 3])
        tags: Dictionary mapping tag strings to IDs (e.g., {"B-PER": 4, "O": 0})

    Returns:
        List of tuples: (chunk_type, chunk_start, chunk_end)
    """
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None

    for i, tok in enumerate(seq):
        tag = idx_to_tag.get(tok, 'O')
        if tag == 'O':
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = None, None
            continue

        parts = tag.split('-')
        tag_class = parts[0]
        tag_type = parts[1] if len(parts) > 1 else tag_class

        if chunk_type is not None:
            if tag_class in ['B', 'S'] or tag_type != chunk_type:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = None, None

        if chunk_type is None:
            chunk_type, chunk_start = tag_type, i

        if tag_class in ['E', 'S']:
            chunks.append((chunk_type, chunk_start, i + 1))
            chunk_type, chunk_start = None, None

    if chunk_type is not None:
        chunks.append((chunk_type, chunk_start, len(seq)))
    return chunks


def prepare_char_inputs(chars2, device, char_mode='CNN'):
    """
    Handles character padding and sorting for both CNN and LSTM modes for NER Task.
    """
    chars2_length = [len(c) for c in chars2]
    char_maxl = max(chars2_length)

    if char_mode == 'LSTM':
        chars2_sorted = sorted(
            enumerate(chars2), key=lambda x: len(x[1]), reverse=True)
        d = {i: original_idx for i,
             (original_idx, _) in enumerate(chars2_sorted)}

        sorted_lengths = [len(c) for _, c in chars2_sorted]
        chars2_mask = torch.zeros(
            (len(chars2_sorted), char_maxl), dtype=torch.long)
        for i, (_, c) in enumerate(chars2_sorted):
            chars2_mask[i, :len(c)] = torch.LongTensor(c)
        return chars2_mask.to(device), sorted_lengths, d
    else:
        d = {}
        chars2_mask = torch.zeros((len(chars2), char_maxl), dtype=torch.long)
        for i, c in enumerate(chars2):
            chars2_mask[i, :len(c)] = torch.LongTensor(c)
        return chars2_mask.to(device), chars2_length, d


def lower_case(x, lower=False):
    if lower:
        return x.lower()
    else:
        return x

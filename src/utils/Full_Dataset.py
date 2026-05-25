import os
import json
import csv
import torch
import multiprocessing as mp
from utils.ner_functions import build_model, predict_ner
from utils.sa_functions import load_sa_model, predict_sa
from collections import OrderedDict
from datasets import load_dataset

base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.normpath(os.path.join(base_dir, ".."))
data_dir = os.path.join(project_dir, "data")
models_dir = os.path.join(project_dir, "models")
src_models_dir = os.path.join(base_dir, "Models")

parameters = OrderedDict()
parameters['train'] = os.path.join(data_dir, "eng.train")
parameters['dev'] = os.path.join(data_dir, "eng.testa")
parameters['test'] = os.path.join(data_dir, "eng.testb")
parameters['tag_scheme'] = "BIOES"
parameters['lower'] = True
parameters['zeros'] = True
parameters['char_dim'] = 30
parameters['word_dim'] = 100
parameters['word_lstm_dim'] = 200
parameters['char_lstm_dim'] = 25
parameters['word_bidirect'] = True
parameters['embedding_path'] = os.path.join(data_dir, "glove.6B.100d.txt")
parameters['all_emb'] = 1
parameters['crf'] = 1
parameters['dropout'] = 0.5
parameters['epoch'] = 50
parameters['weights'] = ""
parameters['name'] = "self-trained-model"
parameters['gradient_clip'] = 5.0
parameters['char_mode'] = "CNN"
models_path = models_dir

# GPU
parameters['use_gpu'] = torch.cuda.is_available()
use_gpu = parameters['use_gpu']
parameters['reload'] = os.path.join(src_models_dir, "pre-trained-model")

# paths to stored mapping file
mapping_file = os.path.join(data_dir, "mapping.pkl")
# To stored model
name = parameters['name']
model_name = os.path.join(models_path, name)

MODEL_SA_NAME = os.path.join(
    models_dir, "sa_models/ConvBiLSTM_v2.0.0_12-04-2026")

# ----- WORD2VEC PRETRAINED MODEL -----
WORD2VEC = "word2vec"
word2vec_url = "https://drive.google.com/uc?id=1zQRH1zYBHJ_vU_uMkKvvvwQiZwP5N7wW"
word2vec_zip = "word2vec.zip"
known_word2vec_gz_file = "GoogleNews-vectors-negative300"
data_dir = "../data"

if not os.path.exists(models_path):
    os.makedirs(models_path)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
SENTINEL = None


def iter_dataset_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            yield item


def reader_worker(dataset_path, input_queue, output_queue, num_gpu_workers):
    base_dir = os.path.dirname(dataset_path)
    for data in iter_dataset_json(dataset_path):
        try:
            if data.get("source") == "bbc":
                continue

            article_subpath = data["article_path"].lstrip("./")
            article_path = os.path.join(base_dir, article_subpath)
            caption = data.get("caption", "")

            with open(article_path, "r", encoding="utf-8") as r:
                article = r.read()

            input_queue.put({
                "article": article,
                "caption": caption,
                "article_path": article_path
            })
        except Exception as e:
            output_queue.put({
                "kind": "reader_error",
                "error": str(e),
                "raw": data
            })

    for _ in range(num_gpu_workers):
        input_queue.put(SENTINEL)


def gpu_worker(input_queue, output_queue, ner_parameters):
    try:
        ner_model, word_to_id, char_to_id, tag_to_id, params, device = build_model(
            ner_parameters)
        sa_model, w2v_model = load_sa_model(MODEL_SA_NAME)
    except Exception as e:
        output_queue.put({
            "kind": "predict_error",
            "error": f"Error inicializando modelo: {e}",
            "raw": {}
        })

        while True:
            item = input_queue.get()
            if item is SENTINEL:
                break

        output_queue.put(SENTINEL)
        return

    while True:
        item = input_queue.get()
        if item is SENTINEL:
            output_queue.put(SENTINEL)
            break

        try:
            ner_output = predict_ner(
                item["article"],
                ner_model,
                word_to_id,
                char_to_id,
                tag_to_id,
                params,
                device
            )

            sa_output = predict_sa(
                item["article"],
                sa_model,
                w2v_model,
                device
            )
            output_queue.put({
                "kind": "ok",
                **item,
                "ner": ner_output,
                "sa": sa_output
            })
        except Exception as e:
            output_queue.put({
                "kind": "predict_error",
                "error": str(e),
                "raw": item
            })


def writer_worker(output_path, output_queue, num_gpu_workers):
    base_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    ok_path = output_path
    reader_err_path = os.path.join(
        base_dir, f"{base_name}_reader_errors.jsonl")
    predict_err_path = os.path.join(
        base_dir, f"{base_name}_predict_errors.jsonl")

    finished = 0
    ok_count = 0
    reader_err_count = 0
    predict_err_count = 0

    ok_example = None
    reader_err_example = None
    predict_err_example = None

    with open(ok_path, "w", encoding="utf-8", newline="") as ok_f, \
            open(reader_err_path, "w", encoding="utf-8") as re_f, \
            open(predict_err_path, "w", encoding="utf-8") as pe_f:

        writer = csv.writer(ok_f, delimiter=";")
        writer.writerow(["article", "image_caption", "NER", "SA"])

        while finished < num_gpu_workers:
            item = output_queue.get()
            if item is SENTINEL:
                finished += 1
                continue

            kind = item.get("kind")
            if kind == "ok":
                writer.writerow([
                    item.get("article", ""),
                    item.get("caption", ""),
                    item.get("ner", ""),
                    item.get("sa", ""),
                ])
                ok_count += 1
                if ok_example is None:
                    ok_example = {
                        "caption": item.get("caption", ""),
                        "ner": str(item.get("ner", ""))[:200]
                    }

            elif kind == "reader_error":
                re_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                reader_err_count += 1
                if reader_err_example is None:
                    reader_err_example = item

            elif kind == "predict_error":
                pe_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                predict_err_count += 1
                if predict_err_example is None:
                    predict_err_example = item

    print(f"[OK] {ok_path} -> {ok_count} filas")
    print(f"[ERR-READER] {reader_err_path} -> {reader_err_count} filas")
    print(f"[ERR-PREDICT] {predict_err_path} -> {predict_err_count} filas")

    print("\nEjemplo OK:")
    print(ok_example if ok_example is not None else "Sin ejemplos")

    print("\nEjemplo ERR-READER:")
    print(reader_err_example if reader_err_example is not None else "Sin ejemplos")

    print("\nEjemplo ERR-PREDICT:")
    print(predict_err_example if predict_err_example is not None else "Sin ejemplos")


def main(dataset_path, output_path, ner_parameters):
    mp.set_start_method("spawn", force=True)

    input_queue = mp.Queue(maxsize=128)
    output_queue = mp.Queue(maxsize=128)

    num_gpu_workers = 1

    reader = mp.Process(target=reader_worker, args=(
        dataset_path, input_queue, output_queue, num_gpu_workers))
    gpu = mp.Process(target=gpu_worker, args=(
        input_queue, output_queue, ner_parameters))
    writer = mp.Process(target=writer_worker, args=(
        output_path, output_queue, num_gpu_workers))

    reader.start()
    gpu.start()
    writer.start()

    reader.join()
    gpu.join()
    writer.join()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(base_dir, "..", "data"))

    os.makedirs(data_dir, exist_ok=True)

    dataset_json = os.path.join(data_dir, "data", "data.json")
    output_csv = os.path.join(data_dir, "clean_dataset.csv")

    if not os.path.exists(dataset_json):
        print(f"Error: No se encontró el archivo de datos en {dataset_json}")
        exit(1)

    main(dataset_json, output_csv, parameters)

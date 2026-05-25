import argparse
import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BlipForConditionalGeneration, BlipProcessor

try:
    from .sa_functions import load_sa_model, predict_sa
    from .finetuning_functions import NewsAlertFineTuner, generate_alert
    from .ner_functions import build_model, normalize_ner_output, predict_ner
except ImportError:
    from sa_functions import load_sa_model, predict_sa
    from finetuning_functions import NewsAlertFineTuner, generate_alert
    from ner_functions import build_model, normalize_ner_output, predict_ner


_THIS_FILE = Path(__file__).resolve()


def _resolve_project_root() -> Path:
    for candidate in _THIS_FILE.parents:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    return _THIS_FILE.parents[2]


PROJECT_ROOT = _resolve_project_root()
SRC_DIR = PROJECT_ROOT / "src"
UTILS_DIR = SRC_DIR / "utils"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

NER_MODEL_PATH = MODELS_DIR / "ner_models" / "best_ner_model.pth"
SA_MODEL_NAME = MODELS_DIR / "sa_models" / "ConvBiLSTM_v2.0.0_12-04-2026"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"

ALERT_MODEL_CANDIDATES = (
    MODELS_DIR / "alert_finetuned_merged",
)
ALERT_MODEL_REPO_ID = "beaotero05/qwen2-Alarm-json"
ALERT_MODEL_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)


def build_ner_parameters() -> OrderedDict:
    parameters = OrderedDict()
    parameters["train"] = str(DATA_DIR / "eng.train")
    parameters["dev"] = str(DATA_DIR / "eng.testa")
    parameters["test"] = str(DATA_DIR / "eng.testb")
    parameters["tag_scheme"] = "BIOES"
    parameters["lower"] = True
    parameters["zeros"] = True
    parameters["char_dim"] = 30
    parameters["word_dim"] = 100
    parameters["word_lstm_dim"] = 200
    parameters["char_lstm_dim"] = 25
    parameters["word_bidirect"] = True
    parameters["embedding_path"] = str(DATA_DIR / "glove.6B.100d.txt")
    parameters["all_emb"] = 1
    parameters["crf"] = 1
    parameters["dropout"] = 0.5
    parameters["epoch"] = 50
    parameters["weights"] = ""
    parameters["name"] = "self-trained-model"
    parameters["gradient_clip"] = 5.0
    parameters["char_mode"] = "CNN"
    parameters["use_gpu"] = torch.cuda.is_available()
    parameters["reload"] = str(NER_MODEL_PATH)
    return parameters


@dataclass
class LoadedNerArtifacts:
    model: Any
    word_to_id: dict[str, int]
    char_to_id: dict[str, int]
    tag_to_id: dict[str, int]
    params: OrderedDict
    device: torch.device


class FineTunedAlertModel:
    def __init__(self, tuner: NewsAlertFineTuner):
        self.tuner = tuner

    def predict(self, entry: dict[str, Any]) -> dict[str, Any]:
        raw = generate_alert(
            model=self.tuner.model,
            tokenizer=self.tuner.tokenizer,
            device=self.tuner.device,
            entry={
                "article": str(entry.get("article", "")),
                "image_caption": str(entry.get("image_caption", "")),
                "NER": str(entry.get("ner", "")),
                "SA": str(entry.get("sa", "")),
            },
        )

        if isinstance(raw, dict) and "error" not in raw:
            return {
                "is_alert": bool(raw.get("is_alert", False)),
                "alert_level": raw.get("alert_level"),
                "category": raw.get("category"),
                "reason": raw.get("reason"),
                "source": "fine_tuned_alert_model",
            }

        raise ValueError(f"Alert model response could not be parsed: {raw}")


@dataclass
class PipelineRequest:
    article_text: Optional[str] = None
    article_path: Optional[str] = None
    image_path: Optional[str] = None
    image_caption: Optional[str] = None
    ner: Optional[str] = None
    sa: Optional[str] = None
    row_number: Optional[int] = None
    row_id: Optional[str] = None


class NewsAlertPipeline:
    def __init__(self, alert_model_path: Optional[str] = None, debug: bool = False):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self._ner: Optional[LoadedNerArtifacts] = None
        self._sa_model: Any = None
        self._sa_w2v_model: Any = None
        self._blip_processor: Optional[BlipProcessor] = None
        self._blip_model: Optional[BlipForConditionalGeneration] = None
        self._alert_model: Optional[Any] = None
        self.alert_model_path = self._resolve_alert_model_path(
            alert_model_path)

    def _find_alert_model_path(self) -> Optional[Path]:
        for candidate in ALERT_MODEL_CANDIDATES:
            if candidate.exists():
                return candidate
        return None

    def _resolve_alert_model_path(
        self, alert_model_path: Optional[str]
    ) -> Optional[str | Path]:
        if alert_model_path:
            expanded_path = Path(alert_model_path).expanduser()
            if expanded_path.exists():
                return expanded_path
            return alert_model_path
        return self._find_alert_model_path()

    def _load_ner(self) -> LoadedNerArtifacts:
        if self._ner is None:
            params = build_ner_parameters()
            model, word_to_id, char_to_id, tag_to_id, loaded_params, device = build_model(
                params,
                debug=self.debug,
            )
            self._ner = LoadedNerArtifacts(
                model=model,
                word_to_id=word_to_id,
                char_to_id=char_to_id,
                tag_to_id=tag_to_id,
                params=loaded_params,
                device=device,
            )
        return self._ner

    def _load_sa(self) -> tuple[Any, Any]:
        if self._sa_model is None or self._sa_w2v_model is None:
            self._sa_model, self._sa_w2v_model = load_sa_model(
                SA_MODEL_NAME,
                debug=self.debug,
            )
        return self._sa_model, self._sa_w2v_model

    def _load_blip(self) -> tuple[BlipProcessor, BlipForConditionalGeneration]:
        if self._blip_processor is None or self._blip_model is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self._blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL_ID,
                dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self._blip_model.eval()
        return self._blip_processor, self._blip_model

    def _load_alert_model(self) -> Any:
        if self._alert_model is not None:
            return self._alert_model

        if self.alert_model_path is None:
            self.alert_model_path = self._download_alert_model(
                ALERT_MODEL_CANDIDATES[0]
            )

        model_source = self._prepare_alert_model_source(self.alert_model_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_source)
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
            )
        except Exception:
            if not isinstance(model_source, Path):
                raise

            synced_model_source = self._download_alert_model(model_source)
            tokenizer = AutoTokenizer.from_pretrained(synced_model_source)
            model = AutoModelForCausalLM.from_pretrained(
                synced_model_source,
                dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
            )
            model_source = synced_model_source

        tuner = NewsAlertFineTuner(model_name=str(model_source))
        tuner.model = model
        tuner.tokenizer = tokenizer
        self._alert_model = FineTunedAlertModel(tuner)
        return self._alert_model

    def _prepare_alert_model_source(self, alert_model_path: str | Path) -> str | Path:
        if isinstance(alert_model_path, Path):
            if self._has_required_alert_files(alert_model_path):
                return alert_model_path
            return self._download_alert_model(alert_model_path)
        return alert_model_path

    def _has_required_alert_files(self, model_dir: Path) -> bool:
        return all((model_dir / filename).exists() for filename in ALERT_MODEL_REQUIRED_FILES)

    def _download_alert_model(self, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=ALERT_MODEL_REPO_ID,
                local_dir=target_dir,
            )
        except Exception as exc:
            if self._has_required_alert_files(target_dir):
                return target_dir
            raise RuntimeError(
                "Failed to download the alert model files from "
                f"`{ALERT_MODEL_REPO_ID}` into `{target_dir}`. "
                "Make sure the machine has internet access and Hugging Face access "
                "to the repository if required."
            ) from exc
        return target_dir

    def caption_image(self, image_path: str) -> str:
        processor, model = self._load_blip()
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
            inputs = processor(
                images=image, return_tensors="pt").to(self.device)
            generated_ids = model.generate(
                **inputs, max_new_tokens=30, num_beams=3)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def run(
        self,
        article_text: Optional[str] = None,
        article_path: Optional[str] = None,
        image_path: Optional[str] = None,
        image_caption: Optional[str] = None,
        ner: Optional[str] = None,
        sa: Optional[str] = None,
    ) -> dict[str, Any]:
        article = self._resolve_article(
            article_text=article_text, article_path=article_path)
        caption = image_caption or (
            self.caption_image(image_path) if image_path else "")

        raw_ner = None
        if ner is not None:
            normalized_ner = ner
        else:
            ner_artifacts = self._load_ner()
            raw_ner = predict_ner(
                article,
                ner_artifacts.model,
                ner_artifacts.word_to_id,
                ner_artifacts.char_to_id,
                ner_artifacts.tag_to_id,
                ner_artifacts.params,
                ner_artifacts.device,
            )
            normalized_ner = normalize_ner_output(raw_ner)

        if sa is not None:
            sentiment = sa
        else:
            sa_model, w2v_model = self._load_sa()
            sentiment = predict_sa(article, sa_model, w2v_model, self.device)

        alert_model = self._load_alert_model()
        alert_input = {
            "article": article,
            "image_caption": caption,
            "ner": normalized_ner,
            "sa": sentiment,
        }
        final_alert = alert_model.predict(alert_input)

        return {
            "article": article,
            "image_path": image_path,
            "image_caption": caption,
            # "ner_raw": raw_ner,
            "ner_normalized": normalized_ner,
            "sentiment": sentiment,
            "final_alert": final_alert,
        }

    def run_batch(self, requests: list[PipelineRequest]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for request in requests:
            item_result: dict[str, Any] = {}
            if request.row_number is not None:
                item_result["row_number"] = request.row_number
            if request.row_id:
                item_result["id"] = request.row_id

            try:
                item_result["result"] = self.run(
                    article_text=request.article_text,
                    article_path=request.article_path,
                    image_path=request.image_path,
                    image_caption=request.image_caption,
                    ner=request.ner,
                    sa=request.sa,
                )
            except Exception as exc:
                item_result["error"] = str(exc)
                item_result["inputs"] = {
                    "article_text": request.article_text,
                    "article_path": request.article_path,
                    "image_path": request.image_path,
                    "image_caption": request.image_caption,
                    "ner": request.ner,
                    "sa": request.sa,
                }

            results.append(item_result)

        return results

    @staticmethod
    def _resolve_article(article_text: Optional[str], article_path: Optional[str]) -> str:
        if article_text and article_text.strip():
            return article_text.strip()
        if article_path:
            return Path(article_path).read_text(encoding="utf-8").strip()
        raise ValueError(
            "You must provide either article_text or article_path.")


def _normalize_optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value: true/false, 1/0, yes/no, on/off."
    )


def _get_row_value(row: dict[str, Any], column_name: Optional[str]) -> Optional[str]:
    if not column_name:
        return None
    if column_name not in row:
        raise ValueError(f"Column `{column_name}` was not found in the CSV.")
    raw_value = row[column_name]
    if raw_value is None:
        return None
    return _normalize_optional_str(str(raw_value))


def _detect_csv_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;|\t")
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else sample
        delimiter_counts = {
            delimiter: first_line.count(delimiter)
            for delimiter in (",", ";", "|", "\t")
        }
        best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        if delimiter_counts[best_delimiter] == 0:
            best_delimiter = ","

        class FallbackDialect(csv.Dialect):
            delimiter = best_delimiter
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return FallbackDialect


def load_batch_requests(
    csv_path: str,
    *,
    article_text_column: Optional[str] = None,
    article_path_column: Optional[str] = None,
    image_path_column: Optional[str] = None,
    image_caption_column: Optional[str] = None,
    ner_column: Optional[str] = None,
    sa_column: Optional[str] = None,
    id_column: Optional[str] = None,
    csv_delimiter: Optional[str] = None,
) -> list[PipelineRequest]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Batch CSV not found: {csv_file}")

    if not article_text_column and not article_path_column:
        raise ValueError(
            "Batch mode requires either `--batch-article-column` or "
            "`--batch-article-path-column`."
        )

    requests: list[PipelineRequest] = []
    with csv_file.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)

        if csv_delimiter:
            class ExplicitDialect(csv.Dialect):
                delimiter = csv_delimiter
                quotechar = '"'
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL

            dialect = ExplicitDialect
        else:
            dialect = _detect_csv_dialect(sample)

        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError("Batch CSV must include a header row.")

        missing_columns = [
            column_name
            for column_name in (
                article_text_column,
                article_path_column,
                image_path_column,
                image_caption_column,
                ner_column,
                sa_column,
                id_column,
            )
            if column_name and column_name not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                "The following requested columns were not found in the CSV: "
                + ", ".join(f"`{column}`" for column in missing_columns)
            )

        for row_number, row in enumerate(reader, start=2):
            article_text = _get_row_value(row, article_text_column)
            article_path = _get_row_value(row, article_path_column)
            image_path = _get_row_value(row, image_path_column)
            image_caption = _get_row_value(row, image_caption_column)
            ner = _get_row_value(row, ner_column)
            sa = _get_row_value(row, sa_column)
            row_id = _get_row_value(row, id_column)

            if article_text is None and article_path is None:
                raise ValueError(
                    "Each batch row must include a value in the configured article column "
                    "or article-path column. "
                    f"Failed at row {row_number}."
                )

            requests.append(
                PipelineRequest(
                    article_text=article_text,
                    article_path=article_path,
                    image_path=image_path,
                    image_caption=image_caption,
                    ner=ner,
                    sa=sa,
                    row_number=row_number,
                    row_id=row_id,
                )
            )

    return requests


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the news article alert pipeline.")
    single_article_group = parser.add_mutually_exclusive_group()
    single_article_group.add_argument("--article", type=str, help="Raw article text.")
    single_article_group.add_argument("--article-path", type=str,
                                      help="Path to a text file containing the article.")
    parser.add_argument("--image-path", type=str,
                        help="Path to the article image.")
    parser.add_argument("--image-caption", type=str,
                        help="Optional caption to use instead of BLIP caption generation.")
    parser.add_argument("--ner", type=str,
                        help="Optional precomputed NER output to use instead of running NER.")
    parser.add_argument("--sa", type=str,
                        help="Optional precomputed sentiment output to use instead of running sentiment analysis.")
    parser.add_argument(
        "--batch-csv",
        type=str,
        help="Path to a CSV file for batch processing.",
    )
    parser.add_argument(
        "--batch-article-column",
        type=str,
        help="CSV column containing raw article text.",
    )
    parser.add_argument(
        "--batch-article-path-column",
        type=str,
        help="CSV column containing paths to article text files.",
    )
    parser.add_argument(
        "--batch-image-column",
        type=str,
        help="CSV column containing image paths.",
    )
    parser.add_argument(
        "--batch-image-caption-column",
        type=str,
        help="CSV column containing precomputed image captions.",
    )
    parser.add_argument(
        "--batch-ner-column",
        type=str,
        help="CSV column containing precomputed NER output.",
    )
    parser.add_argument(
        "--batch-sa-column",
        type=str,
        help="CSV column containing precomputed sentiment output.",
    )
    parser.add_argument(
        "--batch-id-column",
        type=str,
        help="Optional CSV column to copy into the output item id.",
    )
    parser.add_argument(
        "--batch-csv-delimiter",
        type=str,
        choices=[",", ";", "|", "\t"],
        help="Optional CSV delimiter override for batch mode.",
    )
    parser.add_argument(
        "--alert-model-path",
        type=str,
        help="Optional path to a local fine-tuned alert model directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save the JSON output. Defaults to stdout only.",
    )
    parser.add_argument(
        "--stdout",
        type=_parse_bool_arg,
        nargs="?",
        const=True,
        default=False,
        help="Print the JSON output to stdout (default: false).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )
    parser.add_argument(
        "--debug",
        type=_parse_bool_arg,
        nargs="?",
        const=True,
        default=False,
        help="Enable debug logs (default: false).",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.batch_csv and not args.article and not args.article_path:
        parser.error(
            "Single mode requires either --article or --article-path. "
            "Use --batch-csv for batch mode."
        )

    pipeline = NewsAlertPipeline(
        alert_model_path=args.alert_model_path,
        debug=args.debug,
    )
    if args.batch_csv:
        batch_requests = load_batch_requests(
            args.batch_csv,
            article_text_column=args.batch_article_column,
            article_path_column=args.batch_article_path_column,
            image_path_column=args.batch_image_column,
            image_caption_column=args.batch_image_caption_column,
            ner_column=args.batch_ner_column,
            sa_column=args.batch_sa_column,
            id_column=args.batch_id_column,
            csv_delimiter=args.batch_csv_delimiter,
        )
        result: dict[str, Any] = {
            "mode": "batch",
            "input_csv": args.batch_csv,
            "items": pipeline.run_batch(batch_requests),
        }
    else:
        result = {
            "mode": "single",
            "item": pipeline.run(
                article_text=args.article,
                article_path=args.article_path,
                image_path=args.image_path,
                image_caption=args.image_caption,
                ner=args.ner,
                sa=args.sa,
            )
        }

    serialized = (
        json.dumps(result, indent=2, ensure_ascii=False)
        if args.pretty
        else json.dumps(result, ensure_ascii=False)
    )

    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")

    if args.stdout:
        print(serialized)


if __name__ == "__main__":
    main()

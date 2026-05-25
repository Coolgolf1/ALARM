import json
import torch
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")


SYSTEM_PROMPT = (
    "You are a news alert classifier. "
    "Given a news article with its image caption, named entities, and sentiment, "
    "determine if an alert should be triggered. "
    "Always respond with a single valid JSON object and nothing else."
)

USER_TEMPLATE = """\
Article: {article}

Image caption: {image_caption}

Named entities (NER): {ner}

Sentiment: {sa}

Return ONLY a JSON object with this exact structure:
{{
  "is_alert": <true | false>,
  "alert_level": <"critical" | "info" | null>,
  "category": <"brand_safety" | "legal_compliance" | "operational_risk" | "financial_market" | "geopolitical" | "esg_controversy" | "executive_risk" | "reputation_risk" | null>,
  "reason": <"one-sentence explanation" | null>
}}"""


def build_prompt(
    tokenizer: AutoTokenizer,
    article: str,
    image_caption: str,
    ner: str,
    sa: str,
    alert_json: Optional[str] = None,
) -> str:
    """
    Build the chat prompt using the tokenizer's own chat template.

    This ensures the special tokens are always correct for the model being used
    (im_start/end for Qwen, start_of_turn for Gemma, etc.).

    alert_json : include during training to append the target completion.
                 Leave None during inference so the model generates it.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(
            article=article,
            image_caption=image_caption,
            ner=ner,
            sa=sa,
        )},
    ]

    if alert_json is not None:
        messages.append({"role": "assistant", "content": alert_json})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    else:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _extract_first_json(text: str) -> Optional[dict]:
    """
    Extract the first complete, balanced JSON object from text.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def generate_alert(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    entry: dict,
    max_new_tokens: int = 350,
) -> dict:
    """
    Run inference on a single dataset entry.

    Parameters
    ----------
    model, tokenizer : loaded HuggingFace objects
    device           : "cuda" or "cpu"
    entry            : dict with keys article, image_caption, NER, SA

    Returns
    -------
    dict with keys is_alert, alert_level, category, reason
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    prompt = build_prompt(
        tokenizer=tokenizer,
        article=str(entry.get("article", "")),
        image_caption=str(entry.get("image_caption", "")),
        ner=str(entry.get("NER", "")),
        sa=str(entry.get("SA", "")),
        alert_json=None,
    )

    # Keep inputs on the same device as the embedding layer. This prevents
    # CPU/CUDA mismatches when the model was loaded with device_map="auto".
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = torch.device(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    prompt_len = inputs["input_ids"].shape[1]
    closing_brace_id = tokenizer.encode("}", add_special_tokens=False)[-1]

    class _StopOnClosingBrace(StoppingCriteria):
        """Stop generation once the JSON object is balanced (depth reaches 0)."""

        def __init__(self):
            self.depth = 0
            self.started = False

        def __call__(self, input_ids, scores, **kwargs):
            last_token = tokenizer.decode(
                input_ids[0, -1:], skip_special_tokens=True)
            for ch in last_token:
                if ch == "{":
                    self.started = True
                    self.depth += 1
                elif ch == "}" and self.started:
                    self.depth -= 1
                    if self.depth == 0:
                        return True  
            return False

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([_StopOnClosingBrace()]),
        )

    # We decode only the newly generated tokens 
    new_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    parsed = _extract_first_json(response)
    if parsed is not None:
        return {
            "is_alert":    bool(parsed.get("is_alert", False)),
            "alert_level": parsed.get("alert_level"),
            "category":    parsed.get("category"),
            "reason":      parsed.get("reason"),
        }

    print(f"[WARN] Could not parse model response: {response[:200]!r}")
    return {"is_alert": False, "alert_level": None, "category": None, "reason": None}


class NewsAlertFineTuner:
    """QLoRA supervised fine-tuning for news alert classification."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.training_stats = {}

    def load_and_split_data(
        self,
        csv_path: str,
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
        sample_size: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path, sep=";")

        def _is_alert_label(val) -> bool:
            """Return True for alert rows, False for no-alert rows."""
            if not isinstance(val, str):
                return False
            stripped = val.strip()
            if stripped.lower() in {"null", "none", "nan", ""}:
                return False
            try:
                parsed = json.loads(stripped)
                if not isinstance(parsed, dict):
                    return False
                if "is_alert" in parsed:
                    return bool(parsed["is_alert"])
                return any(
                    parsed.get(k) not in (None, "null", "")
                    for k in ("alert_level", "category", "reason")
                )
            except (json.JSONDecodeError, TypeError):
                return True

        df["_is_alert"] = df["Alert"].apply(_is_alert_label)
        n_alerts = df["_is_alert"].sum()
        n_no_alerts = len(df) - n_alerts
        print(
            f"Dataset: {n_alerts} alerts ({n_alerts/len(df):.1%}) + {n_no_alerts} no-alerts ({n_no_alerts/len(df):.1%})")
        df = df.drop(columns=["_is_alert"]).sample(
            frac=1, random_state=42).reset_index(drop=True)

        if sample_size is not None:
            print(f"Using sample of {sample_size} examples")
            df = df.sample(n=min(sample_size, len(df)), random_state=42)

        n_train = int(len(df) * train_ratio)
        n_val = n_train + int(len(df) * val_ratio)

        train_df, val_df, test_df = df[:n_train], df[n_train:n_val], df[n_val:]
        print(
            f"Split — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        return train_df, val_df, test_df

    @staticmethod
    def _normalize_alert_json(raw) -> str:
        """
        Normalize the Alert column value to a consistent JSON string.

        The golden dataset stores alerts as partial JSON (no is_alert key,
        just alert_level/category/reason) and no-alerts as the string "null".
        We convert both to a full JSON with all four expected keys so the model
        always learns to output the same format regardless of the label.
        """
        val = raw if isinstance(raw, str) else str(raw)
        stripped = val.strip()
        if stripped.lower() in {"null", "none", "nan", ""}:
            return json.dumps({
                "is_alert": False,
                "alert_level": None,
                "category": None,
                "reason": None,
            })

        # parse the json output
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                has_content = any(
                    parsed.get(k) not in (None, "null", "")
                    for k in ("alert_level", "category", "reason")
                )
                is_alert = bool(parsed.get("is_alert", has_content))
                return json.dumps({
                    "is_alert":    is_alert,
                    "alert_level": parsed.get("alert_level"),
                    "category":    parsed.get("category"),
                    "reason":      parsed.get("reason"),
                })
        except (json.JSONDecodeError, TypeError):
            pass

        return json.dumps({"is_alert": True, "alert_level": None, "category": None, "reason": None})

    def format_prompts(self, df: pd.DataFrame, is_training: bool = True) -> List[str]:
        prompts = []
        for idx, row in df.iterrows():
            try:
                alert_json = self._normalize_alert_json(
                    row["Alert"]) if is_training else None
                prompts.append(build_prompt(
                    tokenizer=self.tokenizer,
                    article=str(row["article"]),
                    image_caption=str(row["image_caption"]),
                    ner=str(row["NER"]),
                    sa=str(row["SA"]),
                    alert_json=alert_json,
                ))
            except Exception as exc:
                print(f"[WARN] Skipping row {idx}: {exc}")
        print(f"Formatted {len(prompts)} prompts")
        return prompts

    def load_model_and_tokenizer(self):
        print(f"\nLoading {self.model_name} with 4-bit quantization ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" 

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False
        print("Model loaded.")
        return self.model, self.tokenizer

    def setup_lora(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel()
                        for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"LoRA applied — trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        return self.model

    def _tokenize_with_label_masking(self, prompts: List[str]) -> Dataset:
        """
        Tokenize training prompts and set prompt token labels to -100.

        """
        all_input_ids, all_attention_mask, all_labels = [], [], []
        total_answer_tokens = 0
        total_tokens = 0
        for prompt_full in prompts:
            full = self.tokenizer(
                prompt_full,
                truncation=True,
                max_length=2048,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = full["input_ids"][0].tolist()
            attention_mask = full["attention_mask"][0].tolist()

            prompt_only = self._prompt_without_answer(prompt_full)
            prompt_len = len(self.tokenizer(
                prompt_only,
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            )["input_ids"])

            labels = (
                [-100] * prompt_len
                + input_ids[prompt_len:]
            )
            labels = [
                -100 if attention_mask[i] == 0 else labels[i]
                for i in range(len(labels))
            ]
            # ensure length == max_length
            labels = (labels + [-100] * 2048)[:2048]

            answer_tokens = sum(1 for l in labels if l != -100)
            total_answer_tokens += answer_tokens
            total_tokens += sum(attention_mask)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        answer_ratio = total_answer_tokens / max(total_tokens, 1)
        print(
            f"Label masking — answer tokens: {total_answer_tokens:,} / {total_tokens:,} ({answer_ratio:.1%})")
        if answer_ratio < 0.01:
            print(
                "[WARN] Answer ratio is very low — label masking may be broken. Check _prompt_without_answer().")

        return Dataset.from_dict({
            "input_ids":      all_input_ids,
            "attention_mask": all_attention_mask,
            "labels":         all_labels,
        })

    def _prompt_without_answer(self, full_prompt: str) -> str:
        """
        Given a full training prompt (prompt + answer), return the prompt-only
        portion up to and including the assistant turn opening marker.
        This is used to compute the label masking boundary.
        """
        for marker in [
            "<|im_start|>assistant",      
        ]:
            idx = full_prompt.rfind(marker)
            if idx != -1:
                end = full_prompt.find("\n", idx)
                return full_prompt[: end + 1] if end != -1 else full_prompt[:idx]

        return full_prompt

    def fine_tune(
        self,
        train_prompts: List[str],
        val_prompts:   List[str],
        output_dir:    str = "../models/alert-lora",
        num_epochs:    int = 2,
        batch_size:    int = 4,
        learning_rate: float = 5e-5,
    ) -> Trainer:
        print(
            f"\nStarting fine-tuning — {len(train_prompts)} train / {len(val_prompts)} val examples")
        print("Tokenizing with label masking ...")

        train_dataset = self._tokenize_with_label_masking(train_prompts)
        val_dataset = self._tokenize_with_label_masking(val_prompts)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            fp16=True,
            seed=42,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=_LabelPreservingCollator(),
        )
        trainer.train()
        self.training_stats = trainer.state.log_history
        print(f"Fine-tuning complete. Checkpoints saved to {output_dir}")
        return trainer

    def merge_and_save(self, output_dir: str):
        """Merge LoRA weights into the base model and save a standalone HF model."""
        print("Merging LoRA adapters ...")
        merged = self.model.merge_and_unload()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Merged model saved to {output_dir}")


class _LabelPreservingCollator:
    """
    Minimal collator that stacks tensors without touching the labels.
    DataCollatorForLanguageModeling would overwrite our -100 masks.
    """

    def __call__(self, features: List[dict]) -> dict:
        return {
            "input_ids":      torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels":         torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }

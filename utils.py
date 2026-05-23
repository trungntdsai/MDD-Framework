import ast
import json
from typing import Dict, List

import torch
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2FeatureExtractor

import json

SAMPLE_RATE = 16000
PAD_TOKEN_ID = 0
BLANK_TOKEN_ID = 0
ERROR_PAD_ID = 2

with open("/home4/datpt/trungnt/mdd/xlsr_mdd_structured/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

if "" in vocab:
    vocab["<eps>"] = vocab.pop("")

CTC_LABELS = list(vocab.keys())


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_feature_extractor() -> Wav2Vec2FeatureExtractor:
    return Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        padding_side='right',
        do_normalize=True,
        return_attention_mask=False,
    )


def create_decoder():
    return build_ctcdecoder(labels=CTC_LABELS)


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def text_to_tensor(string_text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab[token] for token in string_text.split(' ') if token and token != '$']


def parse_error(error_raw) -> List[int]:
    if isinstance(error_raw, str):
        return ast.literal_eval(error_raw)
    return list(error_raw)


def canonical_time_to_tensor(canonical_time_raw, max_length: int, vocab: Dict[str, int]) -> torch.Tensor:
    canonical_time = torch.full((max_length,), PAD_TOKEN_ID)
    canonical_items = ast.literal_eval(canonical_time_raw) if isinstance(canonical_time_raw, str) else canonical_time_raw

    for item in canonical_items:
        time_span = list(item.keys())[0]
        phone = list(item.values())[0]
        start, end = time_span
        canonical_time[start:end] = vocab[phone]

    return canonical_time


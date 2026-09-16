"""Aligned MOSI/MOSEI caches and independent-length SIMSv2 sequences."""

import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def load_pickle(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Missing dataset file: {}. See docs/data.md; training does not download data.".format(path))
    with path.open("rb") as stream:
        return pickle.load(stream)  # Only load trusted dataset caches.


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest(), "size_bytes": Path(path).stat().st_size}


def _features(value, length, sample_id, name):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError("{}: {} must have a nonempty [time,features] shape".format(sample_id, name))
    if length is not None:
        raw = np.asarray(length)
        if raw.size != 1 or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
            raise ValueError("{}: invalid {} length".format(sample_id, name))
        number = float(raw.reshape(-1)[0])
        if not number.is_integer() or number <= 0 or number > len(value):
            raise ValueError("{}: {} length is outside its sequence".format(sample_id, name))
        value = value[:int(number)]
    if not np.isfinite(value).all():
        raise ValueError("{}: non-finite {} features within the valid sequence".format(sample_id, name))
    return value


class MSADataset(Dataset):
    def __init__(self, config, split, payload=None):
        if split not in {"train", "dev", "test"}:
            raise ValueError("Unknown split: {}".format(split))
        self.config, self.split = config, split
        self.samples = []
        root = Path(config.data_dir)
        if config.data == "simsv2":
            self.path = root / "unaligned.pkl"
            payload = load_pickle(self.path) if payload is None else payload
            key = "valid" if split == "dev" else split
            if key not in payload:
                raise ValueError("SIMSv2 input is missing split {}".format(key))
            source = payload[key]
            required = ("vision", "audio", "raw_text", "regression_labels", "id")
            missing = [key for key in required if key not in source]
            if missing:
                raise ValueError("SIMSv2 split {} missing fields: {}".format(split, missing))
            count = len(source["id"])
            if any(len(source[key]) != count for key in required):
                raise ValueError("SIMSv2 split fields have inconsistent sample counts")
            for field, lengths in (("vision", "vision_lengths"), ("audio", "audio_lengths")):
                dense = isinstance(source[field], np.ndarray) and source[field].ndim == 3
                if dense and lengths not in source:
                    raise ValueError("Dense/padded SIMSv2 {} requires {}".format(field, lengths))
                if lengths in source and len(source[lengths]) != count:
                    raise ValueError("{} count differs from samples".format(lengths))
            for i in range(count):
                vl = source["vision_lengths"][i] if "vision_lengths" in source else None
                al = source["audio_lengths"][i] if "audio_lengths" in source else None
                self._append(source["vision"][i], source["audio"][i], source["raw_text"][i],
                             source["regression_labels"][i], source["id"][i], i, vl, al)
        else:
            self.path = root / (split + ".pkl")
            payload = load_pickle(self.path) if payload is None else payload
            for i, sample in enumerate(payload):
                if not isinstance(sample, (tuple, list)) or len(sample) != 3:
                    raise ValueError("Invalid {} sample {}".format(split, i))
                features, label, video_id = sample
                if len(features) != 4:
                    raise ValueError("Expected (unused, visual, acoustic, words)")
                _, visual, acoustic, words = features
                if not isinstance(words, (list, tuple)) or not all(isinstance(word, str) for word in words):
                    raise ValueError("Cached text must be a list of decoded words")
                if len(visual) != len(acoustic) or len(visual) != len(words):
                    raise ValueError("{} sample {} is not word aligned".format(split, i))
                self._append(visual, acoustic, " ".join(words), label, video_id, i)
        if not self.samples:
            raise ValueError("The {} split is empty".format(split))
        self.visual_size = self.samples[0]["visual"].shape[1]
        self.acoustic_size = self.samples[0]["acoustic"].shape[1]
        for sample in self.samples:
            if sample["visual"].shape[1] != self.visual_size or sample["acoustic"].shape[1] != self.acoustic_size:
                raise ValueError("Inconsistent feature dimensions in {}".format(split))

    def _append(self, visual, acoustic, text, label, original_id, index, visual_length=None, acoustic_length=None):
        sample_id = "{}:{}:{}".format(self.split, index, original_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("{}: missing raw text".format(sample_id))
        label = np.asarray(label, dtype=np.float32).reshape(-1)
        limit = 1 if self.config.data == "simsv2" else 3
        if label.size != 1 or not np.isfinite(label).all() or abs(float(label[0])) > limit + 1e-6:
            raise ValueError("{}: expected one finite label in [{},{}]".format(sample_id, -limit, limit))
        self.samples.append({
            "visual": _features(visual, visual_length, sample_id, "visual"),
            "acoustic": _features(acoustic, acoustic_length, sample_id, "acoustic"),
            "text": text, "label": float(label[0]), "id": sample_id, "source_id": str(original_id),
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class Collator:
    def __init__(self, tokenizer, max_text_length=512):
        self.tokenizer, self.max_text_length = tokenizer, max_text_length

    def __call__(self, samples):
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        tokens = self.tokenizer([s["text"] for s in samples], padding=True, truncation=True,
                                max_length=self.max_text_length, return_tensors="pt")
        batch = {key: tokens[key] for key in ("input_ids", "attention_mask", "token_type_ids") if key in tokens}
        for modality in ("visual", "acoustic"):
            values = [torch.as_tensor(s[modality], dtype=torch.float32) for s in samples]
            batch[modality] = pad_sequence(values, batch_first=True)
            batch[modality + "_lengths"] = torch.tensor([len(value) for value in values], dtype=torch.long)
        batch["labels"] = torch.tensor([s["label"] for s in samples], dtype=torch.float32)
        batch["ids"] = [s["id"] for s in samples]
        batch["source_ids"] = [s["source_id"] for s in samples]
        return batch


def make_loaders(config, splits=("train", "dev", "test"), tokenizer=None):
    payload = load_pickle(Path(config.data_dir) / "unaligned.pkl") if config.data == "simsv2" else None
    datasets = {split: MSADataset(config, split, payload) for split in splits}
    reference = next(iter(datasets.values()))
    for dataset in datasets.values():
        if (dataset.visual_size, dataset.acoustic_size) != (reference.visual_size, reference.acoustic_size):
            raise ValueError("Feature dimensions differ between splits")
    for field, value in (("visual_size", reference.visual_size), ("acoustic_size", reference.acoustic_size)):
        if getattr(config, field) not in (0, value):
            raise ValueError("{} differs from the checkpoint configuration".format(field))
        setattr(config, field, value)
    names = list(datasets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = {s["source_id"] for s in datasets[left].samples} & {s["source_id"] for s in datasets[right].samples}
            if overlap:
                raise ValueError("Source IDs overlap between {} and {}".format(left, right))
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.bert_dir, revision=config.revision)
    collator = Collator(tokenizer, config.max_text_length)
    return {split: DataLoader(dataset, batch_size=config.batch_size, shuffle=split == "train",
                              collate_fn=collator, num_workers=0,
                              generator=torch.Generator().manual_seed(config.seed))
            for split, dataset in datasets.items()}, tokenizer

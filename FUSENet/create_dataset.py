"""Build word-aligned MOSI/MOSEI caches using the CMU Multimodal SDK."""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from .runtime import _atomic_write


RECIPES = {
    "mosi": ("CMU_MOSI_TimestampedWords", "CMU_MOSI_VisualFacet_4.1", "CMU_MOSI_COVAREP", "CMU_MOSI_Opinion_Labels"),
    "mosei": ("CMU_MOSEI_TimestampedWords", "CMU_MOSEI_VisualFacet42", "CMU_MOSEI_COVAREP", "CMU_MOSEI_LabelsSentiment"),
}


def prepare(dataset_name, data_dir, overwrite=False):
    if dataset_name not in RECIPES:
        raise ValueError("Only MOSI/MOSEI use this word-alignment preprocessing")
    root = Path(data_dir).expanduser().resolve()
    cache_paths = [root / (split + ".pkl") for split in ("train", "dev", "test")]
    if any(path.exists() for path in cache_paths) and not overwrite:
        raise FileExistsError("Caches already exist; use --overwrite only for an intentional rebuild")
    try:
        from mmsdk import mmdatasdk as md
    except ImportError as exc:
        raise ImportError("Install CMU-MultimodalSDK to prepare raw data; cached training does not require it") from exc
    root.mkdir(parents=True, exist_ok=True)
    definition = getattr(md, "cmu_" + dataset_name)
    words_name, visual_name, audio_name, label_name = RECIPES[dataset_name]
    remote = {}
    for kind in ("raw", "highlevel", "labels"):
        remote.update(getattr(definition, kind, {}))
    needed = [name for name in RECIPES[dataset_name] if not (root / (name + ".csd")).is_file()]
    unavailable = [name for name in needed if name not in remote]
    if unavailable:
        raise ValueError("The installed SDK has no download recipe for: {}. Supply those .csd files explicitly.".format(unavailable))
    if needed:
        md.mmdataset({name: remote[name] for name in needed}, str(root))
    feature_names = (words_name, visual_name, audio_name)
    dataset = md.mmdataset({name: str(root / (name + ".csd")) for name in feature_names})

    def average(intervals, features):
        return np.average(features, axis=0)

    dataset.align(words_name, collapse_functions=[average])
    dataset.add_computational_sequences({label_name: str(root / (label_name + ".csd"))}, destination=None)
    dataset.align(label_name)
    folds = definition.standard_folds
    membership = {"train": set(folds.standard_train_fold), "dev": set(folds.standard_valid_fold), "test": set(folds.standard_test_fold)}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if membership[left] & membership[right]:
            raise ValueError("SDK video folds are not disjoint")
    samples = {name: [] for name in membership}
    excluded, retained = [], {name: [] for name in membership}
    for segment in dataset[label_name].keys():
        reason = None
        match = re.fullmatch(r"(.*)\[.*\]", segment)
        if not match:
            excluded.append({"segment": segment, "reason": "invalid_segment_id"})
            continue
        video_id = match.group(1)
        split = next((name for name, ids in membership.items() if video_id in ids), None)
        if split is None:
            excluded.append({"segment": segment, "reason": "outside_standard_folds"})
            continue
        try:
            words = dataset[words_name][segment]["features"]
            visual = dataset[visual_name][segment]["features"]
            acoustic = dataset[audio_name][segment]["features"]
            label = np.asarray(dataset[label_name][segment]["features"], dtype=np.float32).reshape(-1)
            if not len(words) == len(visual) == len(acoustic):
                reason = "unaligned_lengths"
            elif label.size != 1 or not np.isfinite(label).all() or abs(float(label[0])) > 3:
                reason = "invalid_target"
            else:
                decoded = [word[0].decode("utf-8") if isinstance(word[0], bytes) else str(word[0]) for word in words]
                keep = [i for i, word in enumerate(decoded) if word != "sp"]
                if not keep:
                    reason = "empty_after_silence_removal"
                else:
                    visual, acoustic = np.asarray(visual[keep], dtype=np.float32), np.asarray(acoustic[keep], dtype=np.float32)
                    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                        visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (1e-6 + visual.std(0, keepdims=True)))
                        acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (1e-6 + acoustic.std(0, keepdims=True)))
                    samples[split].append(((None, visual, acoustic, [decoded[i] for i in keep]), label, video_id))
                    retained[split].append(segment)
        except (KeyError, IndexError, UnicodeError, ValueError) as exc:
            reason = "invalid_segment: {}".format(exc)
        if reason:
            excluded.append({"segment": segment, "reason": reason})
    if any(not values for values in samples.values()):
        raise ValueError("Preprocessing produced an empty split; no cache files were written")
    for split, values in samples.items():
        def write(destination, values=values):
            with destination.open("wb") as stream:
                pickle.dump(values, stream, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_write(root / (split + ".pkl"), write)
    report = {"dataset": dataset_name, "sdk_module": getattr(md, "__file__", None),
              "counts": {k: len(v) for k, v in samples.items()}, "segments": retained, "excluded": excluded}
    _atomic_write(root / "preprocessing.json", lambda path: path.write_text(json.dumps(report, indent=2) + "\n"))
    print(json.dumps({"counts": report["counts"], "excluded": len(excluded)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=tuple(RECIPES), required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.data, args.data_dir, args.overwrite)


if __name__ == "__main__":
    main()

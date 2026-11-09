#!/usr/bin/env python3
"""Evaluate one Claude transition run against MOSCATO ground truth."""

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

import wandb

PROJECT = "claude - full evals"
ENTITY = "chefOST"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


def canonicalize(text, aliases):
    text = text.lower().strip()
    if text in aliases:
        return aliases[text]
    matches = [
        (len(alias), canonical)
        for alias, canonical in aliases.items()
        if alias and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text)
    ]
    return max(matches, default=(0, None))[1]


def mapped_state_set(text, obj, mappings):
    entry = mappings[obj][text.lower().strip()]
    return set(entry["mapped"]) | {f"unmapped:{state}" for state in entry["unmapped"]}


def intervals_and_events(obj, objects, states):
    eligible = [obj in visible and bool(labels) for visible, labels in zip(objects, states)]
    intervals = []
    start = None
    for frame, yes in enumerate(eligible + [False]):
        if yes and start is None:
            start = frame
        elif not yes and start is not None:
            events, previous = [], None
            for index in range(start, frame):
                current = frozenset(states[index])
                if previous is not None and current != previous:
                    events.append(index)
                previous = current
            intervals.append((start, frame, events))
            start = None
    return eligible, intervals


def best_matches(gt, predicted, tolerance):
    # Ordered 1-D matching: maximize matches, then minimize total timing error.
    dp = [[(0, 0.0, []) for _ in range(len(predicted) + 1)] for _ in range(len(gt) + 1)]
    for i in range(1, len(gt) + 1):
        for j in range(1, len(predicted) + 1):
            choices = [dp[i - 1][j], dp[i][j - 1]]
            error = abs(gt[i - 1] - predicted[j - 1])
            if error <= tolerance:
                matches, cost, pairs = dp[i - 1][j - 1]
                choices.append((matches + 1, cost + error, pairs + [(i - 1, j - 1)]))
            dp[i][j] = max(choices, key=lambda item: (item[0], -item[1]))
    return dp[-1][-1]


def best_match_count(gt, predicted, tolerance):
    matches, cost, _ = best_matches(gt, predicted, tolerance)
    return matches, cost


def timing_delta(all_ground_truth, fps_by_video):
    gaps = []
    for video, annotation in all_ground_truth.items():
        fps = fps_by_video[video]
        for obj, timeline in annotation["state"].items():
            _, intervals = intervals_and_events(obj, annotation["object"], timeline)
            for _, _, events in intervals:
                gaps.extend((b - a) / fps for a, b in zip(events, events[1:]))
    if len(gaps) < 2:
        return 0.0, len(gaps)
    mean = sum(gaps) / len(gaps)
    return math.sqrt(sum((gap - mean) ** 2 for gap in gaps) / len(gaps)), len(gaps)


def prediction_timelines(rows, object_aliases, state_mappings, frame_count):
    events = {}
    ignored = []
    for frame, raw_object, before, after in rows:
        obj = canonicalize(raw_object, object_aliases)
        if obj not in state_mappings:
            ignored.append({"frame": frame, "object": raw_object})
            continue
        events.setdefault(obj, []).append((int(frame), before, after))

    timelines = {}
    for obj, object_events in events.items():
        object_events.sort()
        labels = [None] * frame_count
        text = [None] * frame_count
        first_frame, first_before, _ = object_events[0]
        first_frame = min(max(first_frame, 0), frame_count)
        labels[:first_frame] = [mapped_state_set(first_before, obj, state_mappings)] * first_frame
        text[:first_frame] = [first_before] * first_frame
        for index, (frame, _, after) in enumerate(object_events):
            end = object_events[index + 1][0] if index + 1 < len(object_events) else frame_count
            frame, end = min(max(frame, 0), frame_count), min(max(end, 0), frame_count)
            labels[frame:end] = [mapped_state_set(after, obj, state_mappings)] * max(0, end - frame)
            text[frame:end] = [after] * max(0, end - frame)
        timelines[obj] = {
            "labels": labels,
            "text": text,
            "events": [
                (frame, mapped_state_set(after, obj, state_mappings), after)
                for frame, _, after in object_events
            ],
        }
    return timelines, ignored


def cosine_scores(pairs):
    if not pairs:
        return []
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="mps", model_kwargs={"torch_dtype": "float16"})
    texts = sorted({text for pair in pairs for text in pair})
    embeddings = model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    vectors = dict(zip(texts, embeddings))
    return [float(vectors[predicted] @ vectors[truth]) for predicted, truth in pairs]


def evaluate(annotation, rows, object_aliases, state_mappings, delta_seconds, source_fps):
    frame_count = len(annotation["object"])
    predicted, ignored = prediction_timelines(rows, object_aliases, state_mappings, frame_count)
    precision, recall, cosine_pairs, missing_cosine = [], [], [], 0

    gt_events = predicted_events = matches = 0
    for obj in sorted(state_mappings):
        gt_timeline = annotation["state"][obj]
        eligible, intervals = intervals_and_events(obj, annotation["object"], gt_timeline)
        pred = predicted.get(obj)
        for frame, is_eligible in enumerate(eligible):
            if not is_eligible:
                continue
            truth = set(gt_timeline[frame])
            guess = pred["labels"][frame] if pred else None
            if guess:
                overlap = len(guess & truth)
                precision.append(overlap / len(guess))
                recall.append(overlap / len(truth))
                cosine_pairs.append((pred["text"][frame], ", ".join(sorted(truth))))
            else:
                recall.append(0.0)
                missing_cosine += 1

        object_predictions = pred["events"] if pred else []
        assigned = set()
        for start, end, events in intervals:
            local = [event for event in object_predictions if start <= event[0] < end]
            assigned.update(event[0] for event in local)
            match_count, _ = best_match_count(
                [frame / source_fps for frame in events],
                [event[0] / source_fps for event in local],
                delta_seconds,
            )
            gt_events += len(events)
            predicted_events += len(local)
            matches += match_count
        predicted_events += len(object_predictions) - len(assigned)

    cosine = cosine_scores(cosine_pairs)
    tp, fp, fn = matches, predicted_events - matches, gt_events - matches
    return {
        "precision": sum(precision) / len(precision) if precision else None,
        "recall_detail": sum(recall) / len(recall) if recall else None,
        "cosine_similarity": sum(cosine) / (len(cosine) + missing_cosine) if cosine or missing_cosine else None,
        "timing_f1": 2 * tp / (gt_events + predicted_events) if gt_events + predicted_events else None,
        "timing_tp": tp,
        "timing_fp": fp,
        "timing_fn": fn,
        "gt_events": gt_events,
        "predicted_events": predicted_events,
        "precision_frames": len(precision),
        "recall_frames": len(recall),
        "cosine_compared_frames": len(cosine),
        "cosine_missing_frames": missing_cosine,
        "matched_objects": sorted(state_mappings),
        "ignored_object_events": len(ignored),
        "ignored_objects": sorted({item["object"] for item in ignored}),
    }


def evaluate_new_states(annotation, rows, object_aliases, state_mappings, delta_seconds, source_fps):
    predicted, ignored = prediction_timelines(rows, object_aliases, state_mappings, len(annotation["object"]))
    precision, recall, cosine_pairs, missing_cosine = [], [], [], 0
    gt_events = predicted_events = timing_matches = added_gt_events = 0

    for obj in sorted(state_mappings):
        timeline = annotation["state"][obj]
        _, intervals = intervals_and_events(obj, annotation["object"], timeline)
        object_predictions = predicted.get(obj, {}).get("events", [])
        handled_predictions = set()

        for start, end, events in intervals:
            local = [(index, event) for index, event in enumerate(object_predictions) if start <= event[0] < end]
            pairs = best_matches(
                [frame / source_fps for frame in events],
                [event[0] / source_fps for _, event in local],
                delta_seconds,
            )[2]
            matched_gt = {gt_index for gt_index, _ in pairs}
            matched_pred = {pred_index for _, pred_index in pairs}
            handled_predictions.update(index for index, _ in local)
            gt_events += len(events)
            predicted_events += len(local)
            timing_matches += len(pairs)

            for gt_index, pred_index in pairs:
                frame = events[gt_index]
                target = set(timeline[frame]) - set(timeline[frame - 1])
                if not target:
                    continue
                added_gt_events += 1
                _, labels, text = local[pred_index][1]
                guess = {label for label in labels if not label.startswith("unmapped:")}
                overlap = len(guess & target)
                if guess:
                    precision.append(overlap / len(guess))
                recall.append(overlap / len(target))
                cosine_pairs.append((text, ", ".join(sorted(target))))

            for gt_index, frame in enumerate(events):
                if gt_index not in matched_gt and set(timeline[frame]) - set(timeline[frame - 1]):
                    added_gt_events += 1
                    recall.append(0.0)
                    missing_cosine += 1

            for pred_index, (_, labels, _) in enumerate(event for _, event in local):
                if pred_index not in matched_pred and any(not label.startswith("unmapped:") for label in labels):
                    precision.append(0.0)

        for index, (_, labels, _) in enumerate(object_predictions):
            if index not in handled_predictions:
                predicted_events += 1
                if any(not label.startswith("unmapped:") for label in labels):
                    precision.append(0.0)

    cosine = cosine_scores(cosine_pairs)
    tp, fp, fn = timing_matches, predicted_events - timing_matches, gt_events - timing_matches
    return {
        "precision": sum(precision) / len(precision) if precision else None,
        "recall_detail": sum(recall) / len(recall) if recall else None,
        "cosine_similarity": sum(cosine) / (len(cosine) + missing_cosine) if cosine or missing_cosine else None,
        "timing_f1": 2 * tp / (gt_events + predicted_events) if gt_events + predicted_events else None,
        "timing_tp": tp,
        "timing_fp": fp,
        "timing_fn": fn,
        "gt_events": gt_events,
        "predicted_events": predicted_events,
        "added_gt_events": added_gt_events,
        "state_precision_events": len(precision),
        "state_recall_events": len(recall),
        "cosine_compared_events": len(cosine),
        "cosine_missing_events": missing_cosine,
        "matched_objects": sorted(state_mappings),
        "ignored_object_events": len(ignored),
        "ignored_objects": sorted({item["object"] for item in ignored}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    default_data = Path(__file__).resolve().parents[2] / "chefOST-data" / "MOSCATO"
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--state-mapping", type=Path, required=True)
    parser.add_argument("--metric-mode", choices=("framewise", "new-states"), default="framewise")
    parser.add_argument("--name-suffix", default="__eval")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    source = api.run(f"{ENTITY}/{PROJECT}/{args.run_id}")
    dataset, video = source.name.split("__", 1)
    names = {"CMU": ("cmu", "object_dict.json"), "EGTEA": ("egtea", "obj_dict.json")}
    slug, object_file = names[dataset]
    ground_truth_path = args.data / "annotations" / "ground_truth" / dataset / f"gt_annotations_{slug}.json"
    all_ground_truth = json.loads(ground_truth_path.read_text())
    annotation = all_ground_truth[video]
    object_aliases = json.loads((args.data / "vocabulary" / dataset / object_file).read_text())["all2one"]
    mapping_document = json.loads(args.state_mapping.read_text())
    state_mappings = mapping_document["mapping"]
    unknown_objects = set(state_mappings) - set(annotation["state"])
    if unknown_objects:
        raise ValueError(f"mapping contains objects absent from ground truth: {sorted(unknown_objects)}")

    fps_by_video = {}
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        if run.name.startswith(dataset + "__") and not run.name.endswith("__eval"):
            fps_by_video[run.name.split("__", 1)[1]] = float(run.config["source_fps"])
    missing = set(all_ground_truth) - set(fps_by_video)
    if missing:
        raise RuntimeError(f"missing source FPS for: {sorted(missing)}")
    delta, gap_count = timing_delta(all_ground_truth, fps_by_video)

    table_file = next(file for file in source.files() if file.name.endswith(".table.json"))
    with tempfile.TemporaryDirectory() as directory:
        downloaded = table_file.download(directory, replace=True)
        table = json.loads(Path(downloaded.name).read_text())
    rows = [dict(zip(table["columns"], row)) for row in table["data"]]
    ordered_rows = [[row["frame"], row["object"], row["state_before"], row["state_after"]] for row in rows]
    evaluator = evaluate_new_states if args.metric_mode == "new-states" else evaluate
    metrics = evaluator(annotation, ordered_rows, object_aliases, state_mappings, delta, float(source.config["source_fps"]))
    metrics.update({"timing_delta_seconds": delta, "timing_gap_count": gap_count})
    print(json.dumps(metrics, indent=2))

    if not args.no_log:
        group = f"eval-{source.id}"
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=f"{source.name}{args.name_suffix}",
            group=group,
            job_type="evaluation",
            config={
                "source_run_id": source.id,
                "dataset": dataset,
                "video": video,
                "ground_truth": str(ground_truth_path),
                "embedding_model": EMBEDDING_MODEL,
                "embedding_prompt": None,
                "metric_mode": args.metric_mode,
                "comparison_unit": "matched state-change events" if args.metric_mode == "new-states" else "eligible object/frame",
                "state_aggregation": "equal mean across matched state-change events" if args.metric_mode == "new-states" else "equal mean across eligible object/frame units",
                "empty_gt_handling": "exclude from state-content metrics",
                "cosine_aggregation": "equal mean; missing prediction scores zero",
                "object_normalization": "longest whole-phrase match in MOSCATO all2one; unmatched objects ignored",
                "state_normalization": "LLM mapping against alphabetized per-object, per-video GT vocabulary",
                "state_mapping_model": mapping_document["model"],
                "state_mapping_sha256": hashlib.sha256(args.state_mapping.read_bytes()).hexdigest(),
                "unmapped_state_handling": "ignored as out-of-vocabulary" if args.metric_mode == "new-states" else "retained as unique false prediction",
                "semantic_target": "newly added GT states at each change" if args.metric_mode == "new-states" else "complete GT state set",
                "prediction_timeline": "first state_before backfilled; state_after carried to next transition",
                "alignment": "source frame index equals annotation array index",
                "timing_delta_split": f"full {dataset} ground-truth split",
            },
        )
        run.log({key: value for key, value in metrics.items() if not isinstance(value, list)})
        run.summary["matched_objects"] = json.dumps(metrics["matched_objects"])
        run.summary["ignored_objects"] = json.dumps(metrics["ignored_objects"])
        artifact = wandb.Artifact(f"state-mapping-{source.id}", type="state-mapping")
        artifact.add_file(str(args.state_mapping), name="state_mapping.json")
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()

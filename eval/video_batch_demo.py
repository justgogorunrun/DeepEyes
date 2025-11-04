"""Enhanced demo utilities for DeepEyes visual reasoning on images and videos.

This module extends :mod:`eval.eval_vstar` by providing:

* Single-media demo that accepts either an image or a video question.
* Video preprocessing that samples 1 FPS frames (up to 512 frames) and tiles
  them into an 8x8 canvas without resizing individual frames.
* Visualization of every reasoning iteration using the bounding boxes returned
  by the model, annotated with the turn index.
* Batch processing from a JSON manifest with real-time accuracy reporting via
  ``tqdm`` and optional multi-GPU parallelism.

Usage:
    python -m eval.video_batch_demo --input_path path/to/image.jpg \
        --question "What is happening?" --api_key ...

    python -m eval.video_batch_demo --json_path manifest.json \
        --api_key ... --devices 0,1
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import multiprocessing as mp
import os
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Prompt configuration copied from eval_vstar.py to preserve behaviour
# -----------------------------------------------------------------------------

instruction_prompt_system = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>"""

USER_PROMPT_V2 = (
    "\nThink first, call **image_zoom_in_tool** if needed, then answer."
    " Format strictly as:  <think>...</think>  <tool_call>...</tool_call>"
    " (if tools needed)  <answer>...</answer> "
)

instruction_prompt_before = """Question: {question}
Options: {options}
""" + USER_PROMPT_V2

user_prompt = USER_PROMPT_V2

start_token = "<tool_call>"
end_token = "</tool_call>"

# -----------------------------------------------------------------------------
# Image utility helpers (copied from eval_vstar.py)
# -----------------------------------------------------------------------------

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to ``number`` that is divisible by ``factor``."""

    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= ``number`` that is divisible by ``factor``."""

    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= ``number`` that is divisible by ``factor``."""

    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Resize helper to keep areas within model constraints."""

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


# -----------------------------------------------------------------------------
# Encoding helpers
# -----------------------------------------------------------------------------

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image_to_base64(pil_image: Image.Image) -> str:
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# -----------------------------------------------------------------------------
# Video frame processing
# -----------------------------------------------------------------------------

def sample_video_frames(
    video_path: str,
    target_fps: float = 1.0,
    max_frames: int = 512,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """Sample frames uniformly at ``target_fps`` up to ``max_frames`` from a video."""

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: List[Image.Image] = []
    sampled_indices: List[int] = []

    if total_frames == 0:
        cap.release()
        raise RuntimeError(f"Video has zero frames: {video_path}")

    if video_fps <= 0:
        video_fps = target_fps

    frame_interval = max(1, int(round(video_fps / target_fps)))

    idx = 0
    sampled = 0
    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            sampled_indices.append(idx)
            sampled += 1
        idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames sampled from video: {video_path}")

    metadata = {
        "video_fps": float(video_fps),
        "total_frames": total_frames,
        "sampled_indices": sampled_indices,
        "sample_count": len(frames),
    }
    return frames, metadata


def tile_frames_to_canvas(frames: Sequence[Image.Image], grid_size: int = 8) -> Image.Image:
    """Tile frames into a ``grid_size`` x ``grid_size`` canvas without resizing."""

    if not frames:
        raise ValueError("No frames provided for tiling")

    frame_width, frame_height = frames[0].size
    cols = grid_size
    rows = int(math.ceil(len(frames) / cols))

    canvas = Image.new("RGB", (frame_width * cols, frame_height * rows), color=(0, 0, 0))

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        canvas.paste(frame, (col * frame_width, row * frame_height))

    return canvas


# -----------------------------------------------------------------------------
# Visualization utilities
# -----------------------------------------------------------------------------

COLORS = [
    (255, 99, 71),
    (135, 206, 250),
    (124, 252, 0),
    (238, 130, 238),
    (255, 215, 0),
    (255, 140, 0),
]


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def draw_bbox_with_label(
    image: Image.Image,
    bbox: Sequence[float],
    label: str,
    color: Tuple[int, int, int],
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = clamp(x1, 0, img.width - 1)
    x2 = clamp(x2, x1 + 1, img.width)
    y1 = clamp(y1, 0, img.height - 1)
    y2 = clamp(y2, y1 + 1, img.height)

    stroke = max(2, int(min(img.size) * 0.004))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    padding = 4
    max_x = max(0, img.width - text_w - 2 * padding)
    max_y = max(0, img.height - text_h - 2 * padding)
    text_x = clamp(x1, 0, max_x)
    text_y = clamp(y1 - text_h - 2 * padding, 0, max_y)
    rect_coords = [text_x, text_y, text_x + text_w + 2 * padding, text_y + text_h + 2 * padding]
    draw.rectangle(rect_coords, fill=color)
    draw.text((text_x + padding, text_y + padding), label, fill=(0, 0, 0), font=font)
    return img


def visualize_turns(
    base_image: Image.Image,
    turn_records: Sequence["TurnRecord"],
    output_dir: str,
    prefix: str,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []
    color_cycle = len(COLORS)
    for idx, record in enumerate(turn_records):
        if record.bbox is None:
            continue
        color = COLORS[idx % color_cycle]
        vis = draw_bbox_with_label(base_image, record.bbox, str(record.turn_index), color)
        filename = f"{prefix}_turn{record.turn_index:02d}.png"
        path = os.path.join(output_dir, filename)
        vis.save(path)
        saved_paths.append(path)
    return saved_paths


# -----------------------------------------------------------------------------
# Iterative reasoning engine
# -----------------------------------------------------------------------------

abc_map = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}


@dataclass
class TurnRecord:
    turn_index: int
    response: str
    bbox: Optional[List[float]]


@dataclass
class ReasoningResult:
    final_answer: str
    status: str
    turn_records: List[TurnRecord]
    raw_messages: List[Dict[str, Any]]
    visualization_paths: List[str] = field(default_factory=list)
    output_dir: str = ""


def build_option_text(options: Optional[Sequence[str]]) -> str:
    if not options:
        return "(No options provided)"
    lines = []
    for idx, option in enumerate(options, 1):
        prefix = abc_map.get(idx, f"Option {idx}")
        clean = option.strip()
        if re.match(r"^[A-Fa-f]\.\s", clean):
            lines.append(clean)
        else:
            lines.append(f"{prefix}. {clean}")
    return "\n" + "\n".join(lines)


def create_client(api_key: str, api_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=api_url)


def fetch_model_name(client: OpenAI, explicit_name: Optional[str]) -> str:
    if explicit_name:
        return explicit_name
    response = client.models.list()
    if not response.data:
        raise RuntimeError("No models available from the API endpoint")
    return response.data[0].id


def parse_bbox_from_tool_call(action_raw: str) -> Optional[List[float]]:
    try:
        action_list = eval(action_raw)  # noqa: S307 - trusted local usage
    except Exception:
        return None
    if isinstance(action_list, dict):
        arguments = action_list.get("arguments", {})
        bbox = (
            arguments.get("bbox_2d")
            or arguments.get("bbox")
            or arguments.get("bbox2d")
        )
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            if isinstance(bbox[0], (list, tuple)):
                bbox = bbox[0]
            return [float(v) for v in bbox[:4]]
    return None


def iterative_reasoning(
    client: OpenAI,
    model_name: str,
    pil_image: Image.Image,
    prompt: str,
    base64_image: Optional[str] = None,
    max_rounds: int = 16,
) -> Tuple[ReasoningResult, List[Dict[str, Any]]]:
    if base64_image is None:
        base64_image = encode_pil_image_to_base64(pil_image)

    messages = [
        {"role": "system", "content": instruction_prompt_system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    print_messages = [
        {"role": "system", "content": instruction_prompt_system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    response_message = ""
    status = "success"
    turn_records: List[TurnRecord] = []
    turn_idx = 0

    try:
        while "</answer>" not in response_message and turn_idx < max_rounds:
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 10240,
                "stop": ["<|im_end|>\n".strip()],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content
            turn_idx += 1

            print_messages.append({"role": "assistant", "content": response_message})

            bbox: Optional[List[float]] = None
            if start_token in response_message and end_token in response_message:
                action_raw = response_message.split(start_token)[1].split(end_token)[0].strip()
                bbox = parse_bbox_from_tool_call(action_raw)

            turn_records.append(TurnRecord(turn_index=turn_idx, response=response_message, bbox=bbox))

            if bbox is not None:
                left, top, right, bottom = bbox
                crop_w = max(1, int(round(right - left)))
                crop_h = max(1, int(round(bottom - top)))
                cropped_image = pil_image.crop((left, top, right, bottom))
                new_h, new_w = smart_resize(crop_h, crop_w)
                cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                cropped_base64 = encode_pil_image_to_base64(cropped_image)

                content_f = [
                    {"type": "text", "text": "<tool_response>"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_base64}"}},
                    {"type": "text", "text": user_prompt},
                    {"type": "text", "text": "</tool_response>"},
                ]

                messages.extend(
                    [
                        {"role": "assistant", "content": response_message},
                        {"role": "user", "content": content_f},
                    ]
                )

                print_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                            {"type": "text", "text": user_prompt},
                        ],
                    }
                )
            else:
                messages.append({"role": "assistant", "content": response_message})

            if turn_idx >= max_rounds:
                status = "max_rounds"
    except Exception as exc:  # noqa: BLE001
        status = "error"
        response_message = str(exc)

    if "</answer>" in response_message and "<answer>" in response_message:
        final_answer = response_message.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        final_answer = response_message.strip()

    result = ReasoningResult(
        final_answer=final_answer,
        status=status,
        turn_records=turn_records,
        raw_messages=print_messages,
    )
    return result, print_messages


# -----------------------------------------------------------------------------
# Answer parsing helpers
# -----------------------------------------------------------------------------

CHOICE_PATTERN = re.compile(r"\b([A-F])\b", re.IGNORECASE)


def normalize_option(option: str) -> str:
    option = option.strip()
    if len(option) >= 2 and option[1] == ".":
        option = option[2:]
    return option.strip().lower()


def extract_choice(final_text: str, options: Optional[Sequence[str]]) -> Optional[str]:
    if not final_text:
        return None
    text = final_text.strip().upper()
    if len(text) == 1 and text in abc_map.values():
        return text
    match = CHOICE_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    if options:
        normalized_text = final_text.strip().lower()
        normalized_text = normalized_text.replace("option ", "").replace("answer ", "")
        for idx, option in enumerate(options, 1):
            opt_clean = normalize_option(option)
            if opt_clean and opt_clean in normalized_text:
                return abc_map.get(idx)
    return None


def normalize_answer_label(answer: Optional[str], options: Optional[Sequence[str]]) -> Optional[str]:
    if not answer:
        return None
    answer = answer.strip()
    if len(answer) == 1 and answer.upper() in abc_map.values():
        return answer.upper()
    return extract_choice(answer, options)


# -----------------------------------------------------------------------------
# Core processing logic
# -----------------------------------------------------------------------------

@dataclass
class SingleRunConfig:
    media_path: str
    question: str
    options: Optional[List[str]]
    sample_id: str
    media_type: str
    answer: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SingleRunResult:
    config: SingleRunConfig
    reasoning: ReasoningResult
    accuracy: Optional[bool]
    pred_choice: Optional[str]
    gold_choice: Optional[str]
    metadata: Dict[str, Any]


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def process_single_media(
    config: SingleRunConfig,
    client: OpenAI,
    model_name: str,
    output_root: str,
    max_video_frames: int = 512,
    max_rounds: int = 16,
) -> SingleRunResult:
    os.makedirs(output_root, exist_ok=True)

    if config.media_type == "video":
        frames, metadata = sample_video_frames(config.media_path, max_frames=max_video_frames)
        base_image = tile_frames_to_canvas(frames, grid_size=8)
        metadata.update({
            "media_type": "video",
            "frame_width": frames[0].width,
            "frame_height": frames[0].height,
        })
    else:
        base_image = Image.open(config.media_path).convert("RGB")
        metadata = {
            "media_type": "image",
            "frame_width": base_image.width,
            "frame_height": base_image.height,
        }

    metadata.update(config.extra)

    prompt = instruction_prompt_before.format(
        question=config.question,
        options=build_option_text(config.options),
    )

    base64_image = encode_pil_image_to_base64(base_image)
    reasoning_result, raw_messages = iterative_reasoning(
        client,
        model_name,
        base_image,
        prompt,
        base64_image,
        max_rounds=max_rounds,
    )

    sample_output_dir = os.path.join(output_root, sanitize_name(config.sample_id))
    os.makedirs(sample_output_dir, exist_ok=True)

    vis_paths = visualize_turns(base_image, reasoning_result.turn_records, sample_output_dir, sanitize_name(config.sample_id))
    reasoning_result.visualization_paths = vis_paths
    reasoning_result.output_dir = sample_output_dir

    messages_path = os.path.join(sample_output_dir, f"{sanitize_name(config.sample_id)}_messages.json")
    with open(messages_path, "w", encoding="utf-8") as f:
        json.dump(raw_messages, f, ensure_ascii=False, indent=2)
    reasoning_result.visualization_paths.append(messages_path)

    pred_choice = extract_choice(reasoning_result.final_answer, config.options)
    gold_choice = normalize_answer_label(config.answer, config.options)
    accuracy = None
    if pred_choice is not None and gold_choice is not None:
        accuracy = pred_choice == gold_choice

    result = SingleRunResult(
        config=config,
        reasoning=reasoning_result,
        accuracy=accuracy,
        pred_choice=pred_choice,
        gold_choice=gold_choice,
        metadata=metadata,
    )
    return result


# -----------------------------------------------------------------------------
# Batch helpers
# -----------------------------------------------------------------------------

def load_json_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # try jsonl
    records: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def detect_media_type(path: str, override: Optional[str] = None) -> str:
    if override:
        return override
    ext = os.path.splitext(path)[1].lower()
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return "video"
    return "image"


def build_config_from_entry(entry: Dict[str, Any], args: argparse.Namespace) -> SingleRunConfig:
    options = entry.get("options")
    question = entry.get("question")
    if question is None:
        raise ValueError("Entry missing 'question'")

    answer = entry.get("answer")
    video_id = entry.get("videoID") or entry.get("video_id")
    image_path = entry.get("image") or entry.get("image_path")

    if image_path:
        media_path = image_path
        media_type = "image"
    elif video_id:
        media_path = os.path.join(args.video_root, f"{video_id}.mp4")
        media_type = "video"
    else:
        raise ValueError("Entry must contain either 'image_path' or 'videoID'")

    question_id = entry.get("question_id") or entry.get("id") or "sample"
    sample_id = f"{video_id or os.path.basename(media_path)}_{question_id}"

    extra = {
        "video_id": video_id,
        "question_id": question_id,
        "task_type": entry.get("task_type"),
        "duration": entry.get("duration"),
        "domain": entry.get("domain"),
        "sub_category": entry.get("sub_category"),
        "source_url": entry.get("url"),
    }

    return SingleRunConfig(
        media_path=media_path,
        question=question,
        options=options,
        sample_id=sample_id,
        media_type=media_type,
        answer=answer,
        extra={k: v for k, v in extra.items() if v is not None},
    )


def worker_process(
    entries: List[Dict[str, Any]],
    args: argparse.Namespace,
    device: Optional[str],
    model_name: str,
    result_queue: mp.Queue,
):
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    client = create_client(args.api_key, args.api_url)
    for entry in entries:
        try:
            config = build_config_from_entry(entry, args)
            result = process_single_media(
                config,
                client,
                model_name,
                args.output_dir,
                max_video_frames=args.max_frames,
                max_rounds=args.max_rounds,
            )
            data = {
                "sample_id": config.sample_id,
                "media_path": config.media_path,
                "question": config.question,
                "final_answer": result.reasoning.final_answer,
                "status": result.reasoning.status,
                "pred_choice": result.pred_choice,
                "gold_choice": result.gold_choice,
                "accuracy": result.accuracy,
                "output_dir": result.reasoning.output_dir,
                "visualizations": result.reasoning.visualization_paths,
                "metadata": result.metadata,
            }
            result_queue.put({"type": "result", "data": data})
        except Exception as exc:  # noqa: BLE001
            data = {
                "sample_id": entry.get("question_id") or entry.get("videoID") or str(time.time()),
                "media_path": entry.get("videoID") or entry.get("image_path"),
                "question": entry.get("question"),
                "final_answer": None,
                "status": f"error: {exc}",
                "pred_choice": None,
                "gold_choice": None,
                "accuracy": False,
                "output_dir": None,
                "visualizations": [],
                "metadata": entry,
            }
            result_queue.put({"type": "result", "data": data})
    result_queue.put({"type": "done"})


def split_entries(entries: List[Dict[str, Any]], num_parts: int) -> List[List[Dict[str, Any]]]:
    parts: List[List[Dict[str, Any]]] = [[] for _ in range(num_parts)]
    for idx, entry in enumerate(entries):
        parts[idx % num_parts].append(entry)
    return [part for part in parts if part]


def parse_devices(device_string: Optional[str]) -> List[Optional[str]]:
    if not device_string:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count > 0:
            return [str(i) for i in range(gpu_count)]
        return [None]
    devices = []
    for token in device_string.split(','):
        token = token.strip()
        if not token:
            continue
        devices.append(token)
    return devices or [None]


def run_batch(args: argparse.Namespace) -> str:
    entries = load_json_manifest(args.json_path)
    if not entries:
        raise ValueError("No entries loaded from JSON manifest")

    client = create_client(args.api_key, args.api_url)
    model_name = fetch_model_name(client, args.eval_model_name)

    devices = parse_devices(args.devices)
    num_workers = len(devices)

    entry_splits = split_entries(entries, num_workers)
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    workers = []
    for device, split in zip(devices, entry_splits):
        p = ctx.Process(target=worker_process, args=(split, args, device, model_name, result_queue))
        p.start()
        workers.append(p)

    total = len(entries)
    finished_workers = 0
    processed = 0
    acc_count = 0
    acc_total = 0
    results: List[Dict[str, Any]] = []

    with tqdm(total=total, desc="Batch inference") as pbar:
        while finished_workers < len(workers):
            item = result_queue.get()
            if item["type"] == "result":
                data = item["data"]
                results.append(data)
                processed += 1
                accuracy = data.get("accuracy")
                if accuracy is not None:
                    acc_total += 1
                    acc_count += int(bool(accuracy))
                postfix = {"acc": f"{(acc_count / acc_total):.3f}" if acc_total else "n/a"}
                pbar.update(1)
                pbar.set_postfix(postfix)
            elif item["type"] == "done":
                finished_workers += 1

    for p in workers:
        p.join()

    run_name = args.run_name or args.model_tag or model_name.replace("/", "_")
    output_path = os.path.join(args.output_dir, f"batch_results_{run_name}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def run_single(args: argparse.Namespace) -> SingleRunResult:
    if not args.input_path:
        raise ValueError("--input_path is required for single run")
    if not args.question:
        raise ValueError("--question is required for single run")

    options = None
    if args.options:
        options = json.loads(args.options)
        if not isinstance(options, list):
            raise ValueError("--options must be a JSON list")

    client = create_client(args.api_key, args.api_url)
    model_name = fetch_model_name(client, args.eval_model_name)

    media_type = detect_media_type(args.input_path, args.media_type)
    sample_id = args.sample_id or os.path.splitext(os.path.basename(args.input_path))[0]
    config = SingleRunConfig(
        media_path=args.input_path,
        question=args.question,
        options=options,
        sample_id=sample_id,
        media_type=media_type,
        answer=args.answer,
        extra={"user_note": args.note} if args.note else {},
    )

    result = process_single_media(
        config,
        client,
        model_name,
        args.output_dir,
        max_video_frames=args.max_frames,
        max_rounds=args.max_rounds,
    )
    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepEyes image/video demo script")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for the inference endpoint")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/v1", help="API URL for the inference endpoint")
    parser.add_argument("--eval_model_name", type=str, default=None, help="Explicit model name for evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs/demo", help="Directory to store results")
    parser.add_argument("--max_frames", type=int, default=512, help="Maximum number of frames sampled from video")
    parser.add_argument("--max_rounds", type=int, default=16, help="Maximum reasoning rounds")
    parser.add_argument("--video_root", type=str, default="/remote-home/zhangkc/data/zhangkc/video-mme-bench/data/", help="Root directory containing videos")
    parser.add_argument("--devices", type=str, default=None, help="Comma separated GPU ids for batch mode")
    parser.add_argument("--json_path", type=str, default=None, help="Path to JSON manifest for batch processing")
    parser.add_argument("--input_path", type=str, default=None, help="Path to a single image or video for demo")
    parser.add_argument("--question", type=str, default=None, help="Question for single demo mode")
    parser.add_argument("--options", type=str, default=None, help="JSON list of options for single demo mode")
    parser.add_argument("--answer", type=str, default=None, help="Ground truth answer for single demo mode")
    parser.add_argument("--media_type", type=str, default=None, choices=["image", "video"], help="Override media type detection")
    parser.add_argument("--sample_id", type=str, default=None, help="Custom sample identifier for single demo")
    parser.add_argument("--note", type=str, default=None, help="Additional note stored in metadata for single demo")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for batch result file")
    parser.add_argument("--model_tag", type=str, default=None, help="Tag appended to batch result filename")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.json_path:
        output_path = run_batch(args)
        print(f"Batch processing completed. Results saved to {output_path}")
    else:
        result = run_single(args)
        print("Single sample processing completed.")
        print(f"Sample ID: {result.config.sample_id}")
        print(f"Final answer: {result.reasoning.final_answer}")
        if result.pred_choice:
            print(f"Predicted choice: {result.pred_choice}")
        if result.gold_choice:
            print(f"Ground truth: {result.gold_choice}")
        if result.accuracy is not None:
            print(f"Accuracy: {result.accuracy}")
        print(f"Outputs saved to: {result.reasoning.output_dir}")


if __name__ == "__main__":
    main()

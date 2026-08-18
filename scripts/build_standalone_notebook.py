"""Build the fully self-contained Online-SDFT demonstration notebook and GIF."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import zlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NOTEBOOK = ROOT / "online_sdft_bandit_demo.ipynb"
GIF_PATH = ROOT / "figures" / "online_sdft_process.gif"
REFERENCE_OUTPUT_DIR = ROOT / "outputs" / "bandit"
REFERENCE_ARTIFACT_NAMES = (
    "per_seed_metrics.csv",
    "summary.json",
    "qualitative_examples.json",
)
CANONICAL_MODEL_REVISION = "13a53837c4906b4f7405932532ba85d182bb013b"
CANONICAL_RUNTIME = {
    "python": "3.11.9",
    "numpy": "2.4.6",
    "torch": "2.13.0",
    "transformers": "5.13.1",
    "peft": "0.19.1",
    "device": "mps",
    "dtype": "float32",
}


def _canonical_event_record(event) -> dict:
    """Serialize every Event field without losing a floating-point bit."""
    return {
        "event_id": event.event_id,
        "phase": event.phase,
        "category": event.category,
        "scenario_id": event.scenario_id,
        "scenario_tier": event.scenario_tier,
        "title": event.title,
        "body": event.body,
        "hour": event.hour.hex(),
        "useful_horizon_minutes": event.useful_horizon_minutes,
        "importance": event.importance.hex(),
        "deadline": event.deadline.hex(),
        "affinity": event.affinity.hex(),
        "busy": event.busy.hex(),
        "x": [float(value).hex() for value in event.x],
        "z": {
            key: float(value).hex()
            for key, value in sorted(event.z.items())
        },
        "sampled_preference": event.sampled_preference,
    }


def canonical_stream_bundle(
    seeds: tuple[int, ...],
    expected_fingerprint: str,
) -> dict:
    """Build the canonical Event payload only in the audited NumPy runtime."""
    import numpy as np

    from online_sdft.environment import DEFAULT_ENVIRONMENT

    if np.__version__ != CANONICAL_RUNTIME["numpy"]:
        raise RuntimeError(
            "canonical stream bundle requires NumPy "
            f"{CANONICAL_RUNTIME['numpy']}, got {np.__version__}"
        )
    live_fingerprint = DEFAULT_ENVIRONMENT.stream_fingerprint(seeds)
    if live_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "live canonical stream fingerprint does not match the tracked "
            f"summary: {live_fingerprint} != {expected_fingerprint}"
        )

    document = {
        "format": "online-sdft-event-stream-v1-float-hex",
        "seeds": [
            {
                "seed": seed,
                "events": [
                    _canonical_event_record(event)
                    for event in DEFAULT_ENVIRONMENT.make_stream(seed)
                ],
            }
            for seed in seeds
        ],
    }
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    event_count = sum(len(entry["events"]) for entry in document["seeds"])
    return {
        "format": document["format"],
        "event_count": event_count,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "zlib_base64": base64.b64encode(
            zlib.compress(payload, level=9)
        ).decode("ascii"),
    }


def canonical_reproduction_manifest() -> dict:
    """Return the audited runtime, source, and exact compact result bytes."""
    summary = json.loads((REFERENCE_OUTPUT_DIR / "summary.json").read_text())
    seeds = (0, 1, 2)
    dataset_fingerprint = summary["config"]["dataset_fingerprint"]
    artifacts = {
        name: (REFERENCE_OUTPUT_DIR / name).read_bytes()
        for name in REFERENCE_ARTIFACT_NAMES
    }
    return {
        "seeds": seeds,
        "model_revision": CANONICAL_MODEL_REVISION,
        "runtime": CANONICAL_RUNTIME,
        "dataset_fingerprint": dataset_fingerprint,
        "stream_bundle": canonical_stream_bundle(
            seeds,
            dataset_fingerprint,
        ),
        "source_fingerprint": summary["config"]["source_fingerprint"],
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in artifacts.items()
        },
        "artifact_zlib_base64": {
            name: base64.b64encode(zlib.compress(payload, level=9)).decode(
                "ascii"
            )
            for name, payload in artifacts.items()
        },
    }


def reproduction_manifest_source() -> str:
    """Render the immutable canonical reference as self-contained Python."""
    return (
        "REFERENCE_ARTIFACT_NAMES = "
        + repr(REFERENCE_ARTIFACT_NAMES)
        + "\nCANONICAL_REPRODUCTION = "
        + repr(canonical_reproduction_manifest())
    )


def embedded_core() -> str:
    """Embed focused modules without any repository imports at runtime."""

    def module_source(
        relative_path: str,
        end_marker: str | None = None,
    ) -> str:
        source = (ROOT / relative_path).read_text()
        if end_marker is not None:
            source = source[:source.index(end_marker)]
        lines = []
        skipping_relative_import = False
        skipping_type_checking = False
        for line in source.splitlines():
            if skipping_type_checking:
                if line and not line[0].isspace():
                    skipping_type_checking = False
                else:
                    continue
            if skipping_relative_import:
                if ")" in line:
                    skipping_relative_import = False
                continue
            if line == "from __future__ import annotations":
                continue
            if line == "from pathlib import Path":
                continue
            if line.startswith("if TYPE_CHECKING:"):
                skipping_type_checking = True
                continue
            stripped = line.lstrip()
            if stripped.startswith("from ."):
                if "(" in stripped and ")" not in stripped:
                    skipping_relative_import = True
                continue
            if line.startswith(("ROOT = ", "OUT = ", "FIG = ")):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    modules = [
        module_source("online_sdft/config.py"),
        module_source("online_sdft/privilege.py"),
        module_source("online_sdft/environment.py"),
        module_source("online_sdft/methods.py"),
        module_source(
            "online_sdft/experiment.py",
            end_marker="\ndef main",
        ),
        module_source(
            "online_sdft/reporting.py",
            end_marker="\n# Frozen baselines",
        ),
    ]
    return "from pathlib import Path\n\n" + "\n\n".join(modules) + "\n"


GIF_FUNCTIONS = r'''
from io import BytesIO
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFont
from matplotlib import font_manager
import matplotlib.pyplot as plt

# The website can render the animation at 64rem (1024 CSS pixels). Draw every
# primitive at 2x density so text and one-pixel rules stay crisp on Retina/HiDPI
# screens instead of asking the browser to enlarge a low-density GIF.
GIF_LOGICAL_WIDTH, GIF_LOGICAL_HEIGHT = 1200, 675
GIF_DRAW_SCALE = 2
GIF_WIDTH = GIF_LOGICAL_WIDTH * GIF_DRAW_SCALE
GIF_HEIGHT = GIF_LOGICAL_HEIGHT * GIF_DRAW_SCALE
PALETTE = {
    "bg": "#F6F5F8",
    "navy": "#1A1A1A",
    "purple": "#7B3F8D",
    "blue": "#4A7FB5",
    "teal": "#4A7A3E",
    "indigo": "#6366F1",
    "coral": "#D97706",
    "slate": "#777777",
    "line": "#E0E0E0",
    "pale_purple": "#F4EDF6",
    "pale_blue": "#EDF3F8",
    "pale_teal": "#EEF3EC",
    "pale_indigo": "#F0F0FD",
    "pale_coral": "#FAF1E6",
    "white": "#FFFFFF",
}


def _font(size, bold=False):
    weight = "bold" if bold else "normal"
    inter_path = Path("figures/fonts/inter/Inter-Variable.ttf")
    if inter_path.exists():
        font = ImageFont.truetype(
            str(inter_path), size=round(size * GIF_DRAW_SCALE)
        )
        try:
            font.set_variation_by_axes([650 if bold else 400])
        except (AttributeError, OSError):
            pass
        return font
    # Preserve a portable fallback when the standalone notebook is copied
    # away from the repository's vendored Inter font.
    path = font_manager.findfont(
        font_manager.FontProperties(
            family=[
                "Inter",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            weight=weight,
        ),
        fallback_to_default=True,
    )
    return ImageFont.truetype(path, size=round(size * GIF_DRAW_SCALE))


FONTS = {
    "title": _font(28, True),
    "kicker": _font(12, True),
    "heading": _font(15, True),
    "body": _font(15),
    "body_bold": _font(15, True),
    "small": _font(13),
    "status": _font(17),
}


def _scaled_coordinates(value):
    """Convert logical drawing coordinates to the high-density canvas."""
    if isinstance(value, tuple):
        return tuple(_scaled_coordinates(item) for item in value)
    if isinstance(value, list):
        return [_scaled_coordinates(item) for item in value]
    return round(value * GIF_DRAW_SCALE)


class _ScaledDraw:
    """ImageDraw facade that keeps the diagram authored in logical pixels."""

    def __init__(self, image):
        self._draw = ImageDraw.Draw(image)

    @staticmethod
    def _scaled_shape_kwargs(kwargs):
        kwargs = dict(kwargs)
        for key in ("radius", "width"):
            if key in kwargs:
                kwargs[key] = max(1, round(kwargs[key] * GIF_DRAW_SCALE))
        return kwargs

    def textbbox(self, xy, text, **kwargs):
        box = self._draw.textbbox(_scaled_coordinates(xy), text, **kwargs)
        return tuple(value / GIF_DRAW_SCALE for value in box)

    def text(self, xy, text, **kwargs):
        self._draw.text(_scaled_coordinates(xy), text, **kwargs)

    def line(self, xy, **kwargs):
        self._draw.line(
            _scaled_coordinates(xy), **self._scaled_shape_kwargs(kwargs)
        )

    def polygon(self, xy, **kwargs):
        self._draw.polygon(_scaled_coordinates(xy), **kwargs)

    def rounded_rectangle(self, xy, **kwargs):
        self._draw.rounded_rectangle(
            _scaled_coordinates(xy), **self._scaled_shape_kwargs(kwargs)
        )

    def ellipse(self, xy, **kwargs):
        self._draw.ellipse(
            _scaled_coordinates(xy), **self._scaled_shape_kwargs(kwargs)
        )

    def rectangle(self, xy, **kwargs):
        self._draw.rectangle(
            _scaled_coordinates(xy), **self._scaled_shape_kwargs(kwargs)
        )


def _rgb(hex_color):
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _blend(foreground, background, alpha):
    fg, bg = _rgb(foreground), _rgb(background)
    return tuple(round(alpha * f + (1 - alpha) * b) for f, b in zip(fg, bg))


def _center(draw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def _arrow(draw, start, end, fill, width=4):
    draw.line((start, end), fill=fill, width=width)
    x, y = end
    direction = 1 if end[0] >= start[0] else -1
    draw.polygon(
        [(x, y), (x - 12 * direction, y - 7), (x - 12 * direction, y + 7)],
        fill=fill,
    )


def _pill(draw, box, text, fill, outline, text_fill, font=None, width=1):
    font = font or FONTS["body_bold"]
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=width)
    _center(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, font, text_fill)


def _card(draw, box, title, intensity, accent, active=False):
    outline = (
        accent if active
        else _blend(PALETTE["line"], PALETTE["bg"], max(0.45, intensity))
    )
    fill = PALETTE["white"] if active else _blend(PALETTE["white"], PALETTE["bg"], intensity)
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=outline,
                           width=2 if active else 1)
    label_color = (
        accent if active
        else _blend(PALETTE["slate"], PALETTE["bg"], intensity)
    )
    _center(draw, ((box[0] + box[2]) / 2, box[1] + 32), title,
            FONTS["heading"], label_color)


def _draw_context(draw, box, intensity):
    color = _blend(PALETTE["purple"], PALETTE["bg"], intensity)
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1 + 55, y1 + 70, x2 - 55, y1 + 155), radius=14,
                           fill=_blend(PALETTE["pale_purple"], PALETTE["bg"], intensity),
                           outline=color, width=2)
    draw.ellipse((x1 + 75, y1 + 91, x1 + 105, y1 + 121), fill=color)
    draw.line((x1 + 115, y1 + 95, x2 - 75, y1 + 95), fill=color, width=5)
    draw.line((x1 + 115, y1 + 116, x2 - 100, y1 + 116), fill=color, width=4)
    labels = [
        "category  calendar", "time  15:00 · weekday",
        "importance  0.88",
    ]
    for index, label in enumerate(labels):
        y = y1 + 170 + index * 34
        _pill(draw, (x1 + 18, y, x2 - 18, y + 26), label,
              _blend(PALETTE["white"], PALETTE["bg"], intensity),
              _blend(PALETTE["line"], PALETTE["bg"], intensity),
              _blend(PALETTE["slate"], PALETTE["bg"], intensity), FONTS["small"])
    _center(draw, ((x1 + x2) / 2, y1 + 348), "student-visible xₜ",
            FONTS["small"], _blend(PALETTE["slate"], PALETTE["bg"], intensity))


def _draw_student(draw, box, intensity):
    x1, y1, x2, _ = box
    labels = [("INTERRUPT", 0.18), ("LATER", 0.67), ("ARCHIVE", 0.15)]
    for index, (label, probability) in enumerate(labels):
        y = y1 + 82 + index * 70
        selected = label == "LATER"
        fill = PALETTE["pale_teal"] if selected else PALETTE["white"]
        outline = PALETTE["teal"] if selected else PALETTE["line"]
        _pill(draw, (x1 + 20, y, x2 - 20, y + 48), label,
              _blend(fill, PALETTE["bg"], intensity),
              _blend(outline, PALETTE["bg"], intensity),
              _blend(PALETTE["navy"], PALETTE["bg"], intensity),
              width=2 if selected else 1)
        draw.rectangle((x1 + 30, y + 55, x1 + 30 + 145 * probability, y + 61),
                       fill=_blend(PALETTE["blue"], PALETTE["bg"], intensity))
        draw.rectangle((x1 + 30 + 145 * probability, y + 55, x2 - 30, y + 61),
                       fill=_blend(PALETTE["line"], PALETTE["bg"], intensity))
    _center(draw, ((x1 + x2) / 2, y1 + 330), "no feedback yet",
            FONTS["small"], _blend(PALETTE["slate"], PALETTE["bg"], intensity))


def _draw_commit(draw, box, intensity):
    x1, y1, x2, _ = box
    _pill(draw, (x1 + 35, y1 + 80, x2 - 35, y1 + 132), "LATER",
          _blend(PALETTE["pale_teal"], PALETTE["bg"], intensity),
          _blend(PALETTE["teal"], PALETTE["bg"], intensity),
          _blend(PALETTE["teal"], PALETTE["bg"], intensity), width=2)
    _center(draw, ((x1 + x2) / 2, y1 + 160), "action is fixed",
            FONTS["small"], _blend(PALETTE["slate"], PALETTE["bg"], intensity))
    for index, label in enumerate(("INTERRUPT · locked", "ARCHIVE · locked")):
        y = y1 + 190 + index * 58
        _pill(draw, (x1 + 22, y, x2 - 22, y + 40), label,
              _blend(PALETTE["pale_coral"], PALETTE["bg"], intensity),
              _blend(PALETTE["coral"], PALETTE["bg"], intensity),
              _blend(PALETTE["coral"], PALETTE["bg"], intensity), FONTS["small"])
    _center(draw, ((x1 + x2) / 2, y1 + 322), "freeze gap before feedback",
            FONTS["small"], _blend(PALETTE["navy"], PALETTE["bg"], intensity))


def _draw_feedback(draw, box, intensity):
    x1, y1, x2, _ = box
    teal = _blend(PALETTE["teal"], PALETTE["bg"], intensity)
    draw.ellipse((x1 + 70, y1 + 80, x2 - 70, y1 + 160),
                 fill=_blend(PALETTE["pale_teal"], PALETTE["bg"], intensity),
                 outline=teal, width=3)
    draw.line((x1 + 91, y1 + 121, x1 + 109, y1 + 139), fill=teal, width=5)
    draw.line((x1 + 109, y1 + 139, x1 + 141, y1 + 102), fill=teal, width=5)
    _center(draw, ((x1 + x2) / 2, y1 + 195), "OPENED LATER",
            FONTS["body_bold"], teal)
    _pill(draw, (x1 + 22, y1 + 228, x2 - 22, y1 + 270), "selection sₜ = LATER",
          _blend(PALETTE["white"], PALETTE["bg"], intensity),
          _blend(PALETTE["line"], PALETTE["bg"], intensity),
          _blend(PALETTE["navy"], PALETTE["bg"], intensity), FONTS["small"])
    _center(draw, ((x1 + x2) / 2, y1 + 315), "hidden preference + unchosen = unknown",
            FONTS["small"], _blend(PALETTE["coral"], PALETTE["bg"], intensity))


def _draw_teacher(draw, box, intensity):
    x1, y1, x2, _ = box
    values = [("I", 0.12), ("L", 0.76), ("A", 0.12)]
    for index, (label, value) in enumerate(values):
        y = y1 + 82 + index * 44
        draw.text((x1 + 25, y), label, font=FONTS["small"],
                  fill=_blend(PALETTE["navy"], PALETTE["bg"], intensity))
        draw.rounded_rectangle((x1 + 50, y, x2 - 25, y + 18), radius=8,
                               fill=_blend(PALETTE["line"], PALETTE["bg"], intensity))
        draw.rounded_rectangle((x1 + 50, y, x1 + 50 + 135 * value, y + 18), radius=8,
                               fill=_blend(PALETTE["teal"], PALETTE["bg"], intensity))
    _center(draw, ((x1 + x2) / 2, y1 + 214), "soft teacher qₜ",
            FONTS["small"], _blend(PALETTE["navy"], PALETTE["bg"], intensity))
    _center(draw, ((x1 + x2) / 2, y1 + 236), "+ observed selection after delay",
            FONTS["small"], _blend(PALETTE["slate"], PALETTE["bg"], intensity))
    for index in range(4):
        left = x1 + 33 + index * 40
        fill = PALETTE["teal"] if index == 0 else PALETTE["blue"]
        draw.rounded_rectangle((left, y1 + 252, left + 28, y1 + 278), radius=6,
                               fill=_blend(fill, PALETTE["bg"], intensity))
    _center(draw, ((x1 + x2) / 2, y1 + 300), "small local batch",
            FONTS["small"], _blend(PALETTE["slate"], PALETTE["bg"], intensity))
    _center(draw, ((x1 + x2) / 2, y1 + 328), "queued for a safe update window",
            FONTS["small"], _blend(PALETTE["teal"], PALETTE["bg"], intensity))


def _draw_frame(progress):
    image = PILImage.new("RGB", (GIF_WIDTH, GIF_HEIGHT), PALETTE["bg"])
    draw = _ScaledDraw(image)

    _center(draw, (GIF_LOGICAL_WIDTH / 2, 28), "ONE CAUSAL ROUND",
            FONTS["kicker"], PALETTE["slate"])
    _center(draw, (GIF_LOGICAL_WIDTH / 2, 59), "How one frozen decision becomes an update",
            FONTS["title"], PALETTE["navy"])
    _center(draw, (GIF_LOGICAL_WIDTH / 2, 91),
            "request · committed action · delayed callback · same-model lesson · future adapter",
            FONTS["body"], PALETTE["slate"])

    left, gap, card_width = 24, 18, 216
    cards = [(left + index * (card_width + gap), 160,
              left + index * (card_width + gap) + card_width, 535)
             for index in range(5)]
    centers = [((box[0] + box[2]) / 2, 128) for box in cards]
    for index in range(4):
        _arrow(draw, (centers[index][0] + 18, 128),
               (centers[index + 1][0] - 18, 128), PALETTE["line"], 3)

    accents = (
        PALETTE["purple"], PALETTE["blue"], PALETTE["indigo"],
        PALETTE["teal"], PALETTE["indigo"],
    )
    current = min(4, int(progress))
    for index, (cx, cy) in enumerate(centers):
        fill = accents[index] if index <= current else PALETTE["line"]
        draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=fill,
                     outline=PALETTE["white"], width=2)

    if progress < 4:
        start = centers[int(progress)][0]
        finish = centers[int(progress) + 1][0]
        fraction = progress - int(progress)
        dot_x = start + (finish - start) * fraction
    else:
        dot_x = centers[-1][0]
    draw.ellipse((dot_x - 5, 123, dot_x + 5, 133), fill=PALETTE["navy"])

    titles = ("1 · Request", "2 · Student", "3 · Commit",
              "4 · Delayed callback", "5 · Teacher + queue")
    renderers = (_draw_context, _draw_student, _draw_commit, _draw_feedback, _draw_teacher)
    for index, box in enumerate(cards):
        if index < current:
            intensity = 0.82
        elif index == current:
            intensity = 1.0
        else:
            intensity = 0.48
        _card(draw, box, titles[index], intensity, accents[index],
              active=index == current)
        renderers[index](draw, box, intensity)

    if current == 4:
        _center(draw, (GIF_LOGICAL_WIDTH / 2, 551), "the LoRA adapter changes only future decisions",
                FONTS["small"], PALETTE["blue"])
        _arrow(draw, (cards[4][2] - 15, 574), (cards[0][0] + 15, 574), PALETTE["blue"], 2)

    status = (
        "Request xₜ arrives; no current feedback exists.",
        "The student chooses an action from Pₜ(a | xₜ).",
        "The selected action is fixed before feedback.",
        "The selected surface exposes a user selection—or UNKNOWN—after its delay.",
        "The same model teaches from that selection; local training waits for a safe window.",
    )[current]
    draw.line((90, 604, 1110, 604), fill=PALETTE["line"], width=1)
    _center(draw, (600, 634), status, FONTS["status"], PALETTE["navy"])
    return image


def make_online_sdft_gif():
    frames = []
    for stage in range(5):
        for substep in range(7):
            progress = stage + substep / 7 if stage < 4 else 4
            frames.append(_draw_frame(progress))
        pause_progress = stage + 0.99 if stage < 4 else 4.0
        frames.extend([_draw_frame(pause_progress)] * 2)
    buffer = BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True,
                   append_images=frames[1:], duration=135, loop=0,
                   optimize=True, disposal=2)
    return buffer.getvalue()
'''.strip()


GAME_ENGINE = r'''
from IPython.display import Markdown, display

GAME_ACTIONS = ("INTERRUPT", "LATER", "ARCHIVE")
GAME_SCENARIOS = (
    {
        "name": "Weekday calendar alert",
        "dataset_version": "semantic-title-body-sharp-t001",
        "sampled_preference": "INTERRUPT",
        "event_id": "s0-p0-0057",
        "title": "Mobile review starts in 10 minutes",
        "body": "Sam asked you to join on time. Tap to open the video call.",
        "category": "calendar", "time": "15:19", "regime": "weekday",
        "importance": 0.9685343635272166,
        "deadline": 0.9941276450659722,
        "affinity": 0.6330453855667049,
        "busy": 0.6530045112295922,
        "incident": 0.0, "manager": 0.0, "social": 0.0,
        "quiet_work": 0.0,
        "useful_horizon_minutes": 10,
    },
    {
        "name": "On-call monitoring incident",
        "dataset_version": "semantic-title-body-sharp-t001",
        "sampled_preference": "INTERRUPT",
        "event_id": "s0-p1-0034",
        "title": "Critical: Identity errors above threshold",
        "body": "Production failures reached 8%. An acknowledgement is requested within 15 minutes.",
        "category": "monitoring", "time": "00:25", "regime": "on-call",
        "importance": 0.9396736910360113,
        "deadline": 0.9966364522020464,
        "affinity": 0.18741322127881177,
        "busy": 0.4304177654866886,
        "incident": 1.0, "manager": 0.0, "social": 0.0,
        "quiet_work": 0.0,
        "useful_horizon_minutes": 15,
    },
    {
        "name": "Off-hours message from a close friend",
        "dataset_version": "semantic-title-body-sharp-t001",
        "sampled_preference": "LATER",
        "event_id": "s0-p2-0061",
        "title": "Maya sent you a message",
        "body": "That trail looks great. Are you free sometime this weekend?",
        "category": "social", "time": "20:11", "regime": "off-hours",
        "importance": 0.4487328928136745,
        "deadline": 0.08691709463106391,
        "affinity": 0.9392671694976312,
        "busy": 0.20804404669712193,
        "incident": 0.0, "manager": 0.0, "social": 0.5,
        "quiet_work": 0.0,
        "useful_horizon_minutes": None,
    },
    {
        "name": "Weekday promotion",
        "dataset_version": "semantic-title-body-sharp-t001",
        "sampled_preference": "ARCHIVE",
        "event_id": "s0-p0-0027",
        "title": "New grocery recommendations selected for you",
        "body": "Browse this week's grocery offers whenever you have time.",
        "category": "promo", "time": "15:03", "regime": "weekday",
        "importance": 0.06147452888802866,
        "deadline": 0.01871778380358485,
        "affinity": 0.0789604719804393,
        "busy": 0.7007237011671852,
        "incident": 0.0, "manager": 0.0, "social": 0.0,
        "quiet_work": 0.0,
        "useful_horizon_minutes": None,
    },
)


def _game_utilities(scenario):
    importance = scenario["importance"]
    affinity = scenario["affinity"]
    urgency = importance * scenario["deadline"]
    busy_cost = 1.20 * scenario["busy"] * (1.0 - 0.65 * urgency)
    interrupt = (1.45 * urgency + 0.42 * affinity - busy_cost
                 + 1.00 * scenario["incident"] + 0.60 * scenario["manager"]
                 + 0.50 * scenario["social"]
                 - 0.65 * scenario["quiet_work"] * (1.0 - urgency))
    later = (0.72 * importance + 0.58 * affinity - 0.62 * urgency
             + 0.22 * scenario["busy"] - 0.62 * scenario["incident"])
    archive = (0.72 * (1 - importance) + 0.36 * (1 - affinity)
               - 0.80 * urgency - 0.50 * scenario["social"])
    horizon = scenario["useful_horizon_minutes"]
    if horizon is not None and horizon < 120:
        missed_fraction = 1.0 - horizon / 120
        stale_digest = archive - missed_fraction * (0.10 + 0.25 * urgency)
        later = min(later, stale_digest)
    return (interrupt, later, archive)


def _execute_game_action(action_index, preference_index):
    if action_index == 0:
        outcomes = (
            ("OPENED_IMMEDIATELY", "INTERRUPT", "MATCH", 1, 5.0),
            ("OPENED_AFTER_DELAY", "LATER", "MISS", 120, -1.0),
            ("DELETED_NOTIFICATION", "ARCHIVE", "MISS", 15, -2.0),
        )
        return outcomes[preference_index]
    if action_index == 1:
        outcomes = (
            ("OPENED_DIGEST", "LATER", "MATCH", 120, 0.25),
            ("OPENED_DIGEST", "LATER", "MATCH", 120, 0.25),
            ("DELETED_FROM_DIGEST", "ARCHIVE", "MISS", 120, -1.0),
        )
        return outcomes[preference_index]
    return ("NO_OBSERVABLE_SELECTION", "UNKNOWN", "UNKNOWN", 240, 0.0)


def play_notification_round(scenario_id, action):
    if not 0 <= scenario_id < len(GAME_SCENARIOS):
        raise ValueError(f"scenario_id must be 0–{len(GAME_SCENARIOS) - 1}")
    action = action.upper()
    if action not in GAME_ACTIONS:
        raise ValueError(f"action must be one of {GAME_ACTIONS}")

    scenario = GAME_SCENARIOS[scenario_id]
    action_index = GAME_ACTIONS.index(action)
    utilities = _game_utilities(scenario)
    best_index = max(range(len(GAME_ACTIONS)), key=lambda index: utilities[index])
    sampled_index = GAME_ACTIONS.index(scenario["sampled_preference"])
    outcome, selection, status, delay, reward = _execute_game_action(
        action_index, sampled_index
    )
    regret = utilities[best_index] - utilities[action_index]

    display(Markdown(f"""
### Your round: {scenario['name']}

| Visible before acting | Value |
| --- | --- |
| Title | **{scenario['title']}** |
| Body | {scenario['body']} |
| Category | `{scenario['category']}` |
| Local time | {scenario['time']} |
| Regime | `{scenario['regime']}` |
| Importance | `{scenario['importance']:.2f}` |
| Dataset row | `{scenario['dataset_version']}` · `{scenario['event_id']}` |

**You committed to:** `{action}`

**One factual outcome:** `{outcome}` after {delay} minute(s)<br>
**Observed user selection:** `{selection}` · `{status}` · shared reward `{reward:+.1f}`

#### Debrief — evaluator only

Current busyness was `{scenario['busy']:.2f}`. The seeded sampled preference was
`{GAME_ACTIONS[sampled_index]}`; the utility-optimal route was
`{GAME_ACTIONS[best_index]}`. This action's regret was `{regret:.3f}` utility
units.

> REINFORCE maps this matured factual outcome to a learner-only reward and
> applies a batched action-token update to its LoRA adapter. That training map
> is distinct from the shared reward displayed above. Online-SDFT instead
> receives a same-LM soft target conditioned on the observed selection. No
> method receives the hidden busyness, preference draw, scoring utilities, or
> optimal route shown in this debrief. `UNKNOWN` is never replaced by either
> evaluator-only answer.
"""))

display(Markdown(
    "✅ **Game engine ready.** Set `SCENARIO_ID` and `MY_ACTION` in the next "
    "cell, then run it."
))
'''.strip()


GAME_PLAY = r'''
# Change these two values, then run this cell.
SCENARIO_ID = 0
MY_ACTION = "INTERRUPT"  # INTERRUPT, LATER, or ARCHIVE

play_notification_round(SCENARIO_ID, MY_ACTION)
'''.strip()


INSTALL_DEPS = r'''
import importlib.metadata as package_metadata
import importlib.util
import os
import subprocess
import sys


in_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
_ONLINE_SDFT_RUNTIME_READY = False
# Colab imports NumPy for its own variable inspector before user cells run.
# Keep that internally consistent numerical stack in Colab; exact numerical
# pins remain the policy for local kernels and the canonical benchmark.
numeric_targets = {} if in_colab else {
    "numpy": ("numpy", "2.4.6"),
}
loaded_conflicts = []
for module_name, (distribution_name, target_version) in numeric_targets.items():
    loaded_module = sys.modules.get(module_name)
    if loaded_module is None:
        continue
    try:
        installed_version = package_metadata.version(distribution_name)
    except package_metadata.PackageNotFoundError:
        installed_version = None
    loaded_version = getattr(loaded_module, "__version__", None)
    if installed_version != target_version or loaded_version != target_version:
        loaded_conflicts.append(
            f"{module_name} loaded={loaded_version}, "
            f"installed={installed_version}, target={target_version}"
        )

# PEFT probes this optional package while dispatching every LoRA target. Colab
# currently preinstalls an old compiled TorchAO that PEFT 0.19 rejects, even
# though this unquantized benchmark never uses TorchAO. Remove the optional
# distribution instead of replacing Colab's CUDA-matched Torch stack.
if in_colab:
    try:
        package_metadata.version("torchao")
    except package_metadata.PackageNotFoundError:
        pass
    else:
        if "torchao" in sys.modules:
            raise RuntimeError(
                "Colab loaded TorchAO before setup. Choose Runtime > "
                "Disconnect and delete runtime, reconnect, and run this "
                "setup cell first."
            )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"]
        )
        importlib.invalidate_caches()

packages = [
    "transformers==5.13.1",
    "peft==0.19.1",
]
if in_colab:
    # These are already present in supported Colab images. Install only a
    # genuinely missing package; never replace a module Colab may have loaded.
    for module_name, requirement in (
        ("numpy", "numpy>=1.17,<3"),
        ("matplotlib", "matplotlib>=3.8,<4"),
        ("PIL", "Pillow>=10,<13"),
    ):
        if importlib.util.find_spec(module_name) is None:
            packages.append(requirement)
else:
    packages.extend(
        [
            "numpy==2.4.6",
            "torch==2.13.0",
            "matplotlib>=3.8,<4",
            "Pillow>=10,<13",
        ]
    )
if in_colab and importlib.util.find_spec("torch") is None:
    # Unusual Colab fallback; normal images already supply CUDA-matched Torch.
    packages.append("torch>=2.4,<3")

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade-strategy",
        "only-if-needed",
        *packages,
    ]
)

# A local notebook may already have loaded a different pinned numerical stack.
# Refuse to mix its native extensions with newly installed files. Colab cannot
# enter this branch because its native numerical stack is deliberately retained.
if loaded_conflicts:
    raise RuntimeError(
        "The live kernel still holds an older numerical stack: "
        f"{'; '.join(loaded_conflicts)}. Restart the Jupyter kernel, then "
        "run this setup cell first."
    )

try:
    import numpy as np
    import numpy.testing  # catches an in-process NumPy upgrade immediately
    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Lfm2Config, Lfm2ForCausalLM

    # Imports alone did not catch Colab's incompatible TorchAO. Construct a
    # tiny LFM with the same Q/K/V/O PEFT LoRA path as the benchmark before
    # declaring the runtime ready; no auxiliary policy implementation is used.
    smoke_base = Lfm2ForCausalLM(
        Lfm2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            block_multiple_of=8,
            full_attn_idxs=[0],
            tie_word_embeddings=False,
        )
    )
    smoke_lora = get_peft_model(
        smoke_base,
        LoraConfig(
            r=1,
            lora_alpha=2,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    assert any(
        "lora_A" in name and parameter.requires_grad
        for name, parameter in smoke_lora.named_parameters()
    )
    with torch.no_grad():
        smoke_lora(input_ids=torch.zeros((1, 2), dtype=torch.long))
    del smoke_lora, smoke_base
except Exception as exc:
    recovery = (
        "In Colab, choose Runtime > Disconnect and delete runtime, reconnect, "
        "and run this corrected setup cell first."
        if in_colab
        else "Create a fresh virtual environment and select it as the kernel."
    )
    raise RuntimeError(
        "Dependency smoke test failed; the live runtime may contain an "
        f"inconsistent numerical or model stack. {recovery}"
    ) from exc

_ONLINE_SDFT_RUNTIME_READY = True

if in_colab:
    if torch.cuda.is_available():
        print(f"GPU runtime detected: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "CPU-only runtime detected. This is fully supported and uses FP32; "
            "the pipeline will take longer than on a T4 GPU."
        )

if torch.cuda.is_available():
    device_name = f"{torch.cuda.get_device_name(0)} (CUDA, FP16)"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device_name = "Apple MPS (FP32)"
else:
    device_name = "CPU (FP32)"
print(
    "Runtime ready | "
    f"python={sys.version.split()[0]} | numpy={np.__version__} | "
    f"torch={torch.__version__} | transformers={transformers.__version__} | "
    f"peft={peft.__version__} | "
    f"device={device_name} | LoRA adapter smoke test=passed"
)
'''.strip()


RUNTIME_GUARD = r'''
if not globals().get("_ONLINE_SDFT_RUNTIME_READY", False):
    raise RuntimeError(
        "Section 5.1 did not finish successfully. Run the setup cell and "
        "follow its recovery instruction before loading or running Online-SDFT."
    )
'''.strip()


RUN_CONFIGURATION = r'''
# Strict mode remains the local Apple-MPS/FP32 default. Colab selects portable
# CUDA/CPU mode automatically because it cannot satisfy the MPS byte gate.
STRICT_BYTE_REPRODUCTION = not bool(globals().get("in_colab", False))
REPRO_OUTPUT_DIR = Path("notebook_reproduction_outputs") / "bandit"
'''.strip()


RUNNER = (
    RUNTIME_GUARD
    + "\n\n"
    + reproduction_manifest_source()
    + "\n\n"
    + r'''
import base64
import hashlib
import importlib.metadata as package_metadata
import sys
import tempfile
import zlib
from io import StringIO
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def _canonical_runtime_status():
    expected = CANONICAL_REPRODUCTION["runtime"]
    actual = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "torch": torch.__version__.split("+")[0],
        "transformers": package_metadata.version("transformers"),
        "peft": package_metadata.version("peft"),
        "device": "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        "dtype": "float32" if torch.backends.mps.is_available() else (
            "float16" if torch.cuda.is_available() else "float32"
        ),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    return actual, mismatches


def _reference_artifact_bytes():
    return {
        name: zlib.decompress(base64.b64decode(encoded))
        for name, encoded in CANONICAL_REPRODUCTION[
            "artifact_zlib_base64"
        ].items()
    }


def _canonical_streams():
    bundle = CANONICAL_REPRODUCTION["stream_bundle"]
    payload = zlib.decompress(base64.b64decode(bundle["zlib_base64"]))
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != bundle["sha256"]:
        raise RuntimeError(
            "embedded canonical stream bundle failed its SHA-256 check: "
            f"{observed_sha256} != {bundle['sha256']}"
        )
    document = json.loads(payload)
    if document["format"] != bundle["format"]:
        raise RuntimeError("embedded canonical stream format does not match")

    streams = {}
    for entry in document["seeds"]:
        seed = int(entry["seed"])
        if seed in streams:
            raise RuntimeError(f"duplicate canonical stream seed: {seed}")
        streams[seed] = [
            Event(
                event_id=record["event_id"],
                phase=int(record["phase"]),
                category=record["category"],
                scenario_id=record["scenario_id"],
                scenario_tier=record["scenario_tier"],
                title=record["title"],
                body=record["body"],
                hour=float.fromhex(record["hour"]),
                useful_horizon_minutes=record["useful_horizon_minutes"],
                importance=float.fromhex(record["importance"]),
                deadline=float.fromhex(record["deadline"]),
                affinity=float.fromhex(record["affinity"]),
                busy=float.fromhex(record["busy"]),
                x=np.asarray(
                    [float.fromhex(value) for value in record["x"]],
                    dtype=float,
                ),
                z={
                    key: float.fromhex(value)
                    for key, value in record["z"].items()
                },
                sampled_preference=record["sampled_preference"],
            )
            for record in entry["events"]
        ]
    if sum(map(len, streams.values())) != bundle["event_count"]:
        raise RuntimeError("embedded canonical stream event count does not match")
    return streams


def _compact_artifact_bytes(config, metrics, rollouts):
    metrics_buffer = StringIO()
    writer = csv.DictWriter(
        metrics_buffer,
        fieldnames=list(metrics[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(metrics)

    with tempfile.TemporaryDirectory(prefix="online-sdft-notebook-") as temp:
        compact_dir = Path(temp)
        write_compact_results(compact_dir, config, metrics, rollouts)
        return {
            "per_seed_metrics.csv": metrics_buffer.getvalue().encode("utf-8"),
            "summary.json": (compact_dir / "summary.json").read_bytes(),
            "qualitative_examples.json": (
                compact_dir / "qualitative_examples.json"
            ).read_bytes(),
        }


def run_canonical_lora_benchmark(strict=True):
    actual_runtime, runtime_mismatches = _canonical_runtime_status()
    if strict and runtime_mismatches:
        details = "; ".join(
            f"{name}: expected {values['expected']}, got {values['actual']}"
            for name, values in runtime_mismatches.items()
        )
        raise RuntimeError(
            "Strict byte reproduction requires the audited MPS/FP32 runtime. "
            + details
            + ". Set STRICT_BYTE_REPRODUCTION=False for a portable, "
            "non-byte-identical run."
        )

    try:
        model_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=CANONICAL_REPRODUCTION["model_revision"],
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        model_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=CANONICAL_REPRODUCTION["model_revision"],
            token=False,
        )
    selected_device = "mps" if strict else "auto"
    seeds = tuple(CANONICAL_REPRODUCTION["seeds"])
    canonical_streams = _canonical_streams()
    if set(canonical_streams) != set(seeds):
        raise RuntimeError("embedded canonical stream seeds do not match")
    curve_fields = [
        "seed", "method", "t", "phase", "regime", "step_correct",
        "step_feedback_reward", "step_regret", "cum_accuracy",
        "cum_regret", "cum_observed_reward",
    ]
    rollout_buffer = StringIO()
    curve_buffer = StringIO()
    curve_writer = csv.DictWriter(
        curve_buffer,
        fieldnames=curve_fields,
        lineterminator="\n",
    )
    curve_writer.writeheader()
    metrics = []
    config = None

    for seed_index, seed in enumerate(seeds, start=1):
        policy = None
        resolved_device = None
        try:
            policy = LiquidLLMPolicy(
                model_id=model_path,
                device=selected_device,
                local_files_only=True,
            )
            resolved_device = str(policy.device)
            if config is None:
                config = experiment_config(
                    seeds=len(seeds),
                    seed_start=seeds[0],
                    model_id=MODEL_ID,
                    policy=policy,
                )
                runtime_dataset_fingerprint = config["dataset_fingerprint"]
                canonical_dataset_fingerprint = (
                    CANONICAL_REPRODUCTION["dataset_fingerprint"]
                )
                if strict:
                    assert runtime_dataset_fingerprint == (
                        canonical_dataset_fingerprint
                    )
                config["dataset_fingerprint"] = canonical_dataset_fingerprint
                config["method_dataset_fingerprints"] = {
                    method: canonical_dataset_fingerprint
                    for method in METHODS
                }
                config["source_fingerprint"] = json.loads(json.dumps(
                    CANONICAL_REPRODUCTION["source_fingerprint"]
                ))
                assert config["methods"] == METHODS
                assert config["online_sdft_trainable_parameters"] == 172032
                assert config["online_rft_trainable_parameters"] == 172032
                assert config["reinforce_trainable_parameters"] == 172032
                print(
                    f"Loaded pinned {MODEL_ID}@"
                    f"{CANONICAL_REPRODUCTION['model_revision'][:12]} on "
                    f"{resolved_device}; six paired methods, common rank-4 LoRA",
                    flush=True,
                )

            stream = canonical_streams[seed]
            for method in METHODS:
                metrics.append(
                    run_method(
                        seed,
                        method,
                        stream,
                        policy,
                        rollout_buffer,
                        curve_writer,
                    )
                )
                print(
                    f"seed {seed_index}/{len(seeds)} (id={seed}) · {method}",
                    flush=True,
                )
                gc.collect()
                if resolved_device == "mps":
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
        finally:
            if policy is not None:
                del policy
            gc.collect()
            if resolved_device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            elif resolved_device == "cuda":
                torch.cuda.empty_cache()

    if config is None:
        raise RuntimeError("canonical seed registry is empty")
    rollouts = [
        json.loads(line)
        for line in rollout_buffer.getvalue().splitlines()
        if line
    ]
    curves = list(csv.DictReader(StringIO(curve_buffer.getvalue())))
    assert len(metrics) == len(seeds) * len(METHODS)
    assert len(rollouts) == len(seeds) * len(METHODS) * STREAM_LENGTH
    assert len(curves) == len(rollouts)

    artifact_bytes = _compact_artifact_bytes(config, metrics, rollouts)
    REPRO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in artifact_bytes.items():
        (REPRO_OUTPUT_DIR / name).write_bytes(payload)
    return {
        "metrics": metrics,
        "rollouts": rollouts,
        "curves": curves,
        "config": config,
        "artifact_bytes": artifact_bytes,
        "reference_bytes": _reference_artifact_bytes(),
        "runtime": actual_runtime,
        "runtime_mismatches": runtime_mismatches,
        "strict": strict,
    }


benchmark = run_canonical_lora_benchmark(STRICT_BYTE_REPRODUCTION)
metrics = benchmark["metrics"]
rollouts = benchmark["rollouts"]
curves = benchmark["curves"]
config = benchmark["config"]
print(
    f"Finished {len(CANONICAL_REPRODUCTION['seeds'])} seeds × "
    f"{len(METHODS)} methods × {STREAM_LENGTH} decisions"
)
'''.strip()
)


RESULTS = r'''
from IPython.display import Markdown, display

if not all(
    name in globals()
    for name in ("benchmark", "metrics", "rollouts", "config")
):
    raise RuntimeError("Run Section 5.4 before displaying results.")

summary = summarize_metrics(metrics)
table = [
    "| Method | Accuracy | Cumulative regret | Reward / decision |",
    "| --- | ---: | ---: | ---: |",
]
for method in METHODS:
    values = summary[method]
    table.append(
        f"| {method} | "
        f"{100 * values['online_accuracy']['mean']:.2f}% ± "
        f"{100 * values['online_accuracy']['ci95']:.2f} | "
        f"{values['cum_regret']['mean']:.2f} ± "
        f"{values['cum_regret']['ci95']:.2f} | "
        f"{values['observed_reward_per_decision']['mean']:.3f} ± "
        f"{values['observed_reward_per_decision']['ci95']:.3f} |"
    )
display(Markdown("\n".join(table)))

assert config["teacher_model"] == MODEL_ID
assert config["student_backbone"] == "frozen Liquid LFM base weights"
assert config["student_backbone_trainable_parameters"] == 0
assert config["online_sdft_trainable_parameters"] == 172032
assert config["online_rft_trainable_parameters"] == 172032
assert config["reinforce_trainable_parameters"] == 172032
assert config["methods"] == METHODS
assert config["teacher_student_model_sharing"]["model_instances"] == 1
assert config["teacher_student_model_sharing"]["student_forward"] == "LoRA adapter enabled"
assert config["teacher_student_model_sharing"]["teacher_forward"] == "same model with LoRA adapter disabled"
assert config["teacher_policy"].startswith("the student and teacher are one shared Liquid LFM model;")
assert "delayed observed user selection" in config["teacher_policy"]
assert "digest open ambiguous between INTERRUPT and LATER" in config["teacher_policy"]
assert "UNKNOWN stays censored" in config["teacher_policy"]
assert "no scalar reward" in config["teacher_policy"]
assert "no end-of-horizon flush" in config["update_timing"]
assert len(metrics) == 3 * len(METHODS)
assert {row["method"] for row in metrics} == set(METHODS)
assert len(rollouts) == 3 * len(METHODS) * STREAM_LENGTH

sdft_rollouts = [
    row for row in rollouts if row["method"] == "Online-SDFT"
]
released_sdft = [
    row for row in rollouts
    if row["method"] == "Online-SDFT"
    and row["feedback_released_at_minute"] is not None
]
censored = [
    row for row in released_sdft
    if row["feedback"]["observed_user_selection"] == "UNKNOWN"
]
applied = [
    row for row in released_sdft
    if row["lesson_status"] == "soft_target_applied"
]
assert sdft_rollouts and released_sdft and censored and applied
assert all(
    row["feedback_released_at_minute"]
    >= row["feedback_available_at_minute"]
    > row["decision_time_minute"]
    for row in released_sdft
)
assert all(
    row["teacher_probs"] is None
    and row["sdft_fusion_weights"] is None
    and row["lesson_status"] == "censored_no_update"
    for row in censored
)


def _first_byte_difference(actual, expected):
    limit = min(len(actual), len(expected))
    for offset in range(limit):
        if actual[offset] != expected[offset]:
            return offset
    return None if len(actual) == len(expected) else limit


artifact_comparison = {}
comparison_table = [
    "| Artifact | Bytes | Generated SHA-256 | Reference SHA-256 | Exact |",
    "| --- | ---: | --- | --- | :---: |",
]
for name in REFERENCE_ARTIFACT_NAMES:
    actual = benchmark["artifact_bytes"][name]
    expected = benchmark["reference_bytes"][name]
    actual_sha = hashlib.sha256(actual).hexdigest()
    expected_sha = hashlib.sha256(expected).hexdigest()
    manifest_sha = CANONICAL_REPRODUCTION["artifact_sha256"][name]
    assert expected_sha == manifest_sha
    exact = actual == expected
    artifact_comparison[name] = {
        "exact": exact,
        "actual_sha256": actual_sha,
        "expected_sha256": expected_sha,
        "first_different_byte": _first_byte_difference(actual, expected),
    }
    comparison_table.append(
        f"| `{name}` | {len(actual):,} | `{actual_sha[:12]}…` | "
        f"`{expected_sha[:12]}…` | {'✅' if exact else '❌'} |"
    )
display(Markdown("\n".join(comparison_table)))

all_exact = all(item["exact"] for item in artifact_comparison.values())
if all_exact:
    display(Markdown(
        "✅ **Byte-identical reproduction passed.** All three tracked compact "
        "artifacts match exactly across all three seeds and all six methods. "
        f"Generated files are in `{REPRO_OUTPUT_DIR}`."
    ))
else:
    details = "; ".join(
        f"{name}: first differing byte "
        f"{values['first_different_byte']}"
        for name, values in artifact_comparison.items()
        if not values["exact"]
    )
    message = "Artifact bytes differ from the audited MPS release: " + details
    if benchmark["strict"]:
        raise AssertionError(message)
    display(Markdown(
        "⚠️ **Portable protocol run completed, but it is not byte-identical.** "
        + message
    ))
'''.strip()


LEARNING_CURVES = r'''
import matplotlib.pyplot as plt

if "curves" not in globals():
    raise RuntimeError("Run Section 5.4 before plotting learning trajectories.")

method_colors = {
    "Base": "#D2D3D6",
    "ICL": "#B7BAC0",
    "RAG": "#969BA4",
    "REINFORCE": "#D97706",
    "RFT": "#75639A",
    "Online-SDFT": "#4A7A3E",
}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
for method in METHODS:
    rows = [row for row in curves if row["method"] == method]
    accuracy_by_step = defaultdict(list)
    regret_by_step = defaultdict(list)
    for row in rows:
        accuracy_by_step[int(row["t"])].append(float(row["cum_accuracy"]))
        regret_by_step[int(row["t"])].append(float(row["cum_regret"]))
    ts = sorted(accuracy_by_step)
    accuracy = [100 * np.mean(accuracy_by_step[t]) for t in ts]
    regret = [np.mean(regret_by_step[t]) for t in ts]
    width = 2.8 if method == "Online-SDFT" else 1.5
    axes[0].plot(ts, accuracy, color=method_colors[method], lw=width, label=method)
    axes[1].plot(ts, regret, color=method_colors[method], lw=width, label=method)

for ax in axes:
    for boundary in (PHASE_LENGTH, 2 * PHASE_LENGTH):
        ax.axvline(boundary, color="#667085", ls="--", lw=1)
    ax.grid(alpha=0.22)
    ax.set_xlabel("Online decisions")
for midpoint, regime in zip(
    (PHASE_LENGTH / 2, 1.5 * PHASE_LENGTH, 2.5 * PHASE_LENGTH),
    REGIMES,
):
    axes[0].text(midpoint, 3, regime, ha="center", va="bottom", color="#475467")
axes[0].set(title="Cumulative online accuracy", ylabel="Accuracy so far (%)", ylim=(0, 100))
axes[1].set(title="Utility gap accumulated in arrival order", ylabel="Cumulative regret")
axes[1].legend(ncol=2, frameon=False)
fig.suptitle("Six-arm paired benchmark · three-seed mean", fontweight="bold")
fig.tight_layout()
plt.show()
'''.strip()


AUDIT_INVARIANT = r'''
from IPython.display import Markdown, display

if "rollouts" not in globals():
    raise RuntimeError("Run Section 5.4 before auditing archived decisions.")

archive_rows = [row for row in rollouts if row["action"] == "ARCHIVE"]
archive_outcomes = Counter(row["feedback"]["outcome"] for row in archive_rows)
assert all("gold_action_scoring_only" in row for row in rollouts)

if archive_rows:
    assert set(archive_outcomes) == {"NO_OBSERVABLE_SELECTION"}
    assert all(row["feedback"]["observed_user_selection"] == "UNKNOWN"
               for row in archive_rows)
    archive_by_method = Counter(row["method"] for row in archive_rows)
    display(Markdown(
        f"**{len(archive_rows):,} archived decisions** produced only factual outcomes: "
        f"`{dict(archive_outcomes)}`. Every archived decision kept the user "
        f"selection UNKNOWN. By method: `{dict(archive_by_method)}`."
    ))
else:
    display(Markdown(
        "ℹ️ This sampled stream did not execute `ARCHIVE`; the branch was not "
        "exercised, so no empirical archive claim is made for this run."
    ))
'''.strip()


ACTION_MIX = r'''
import matplotlib.pyplot as plt

if "rollouts" not in globals():
    raise RuntimeError("Run Section 5.4 before plotting action counts.")

counts = {
    method: Counter(
        row["action"] for row in rollouts if row["method"] == method
    )
    for method in METHODS
}
fig, ax = plt.subplots(figsize=(10.5, 4.8))
action_colors = {"INTERRUPT": "#D95C59", "LATER": "#D9903D", "ARCHIVE": "#667085"}
x = np.arange(len(METHODS))
width = 0.24
for action_index, action in enumerate(ACTIONS):
    values = [counts[method][action] for method in METHODS]
    bars = ax.bar(
        x + (action_index - 1) * width,
        values,
        width,
        label=action,
        color=action_colors[action],
    )
    ax.bar_label(bars, padding=2, fontsize=8)
ax.set_xticks(x, METHODS)
ax.set(title="Executed actions by method · 720 decisions each", ylabel="Decisions")
ax.legend(frameon=False, ncol=3)
ax.grid(axis="y", alpha=0.2)
plt.show()
for method in METHODS:
    print(method + " | " + " | ".join(
        f"{action}={counts[method][action]}" for action in ACTIONS
    ))
'''.strip()


def reader_code_cell(source: str, title: str):
    """Create a code cell whose input is collapsed in Colab/Jupyter viewers."""
    source = f'# @title {title} {{ display-mode: "form" }}\n{source}'
    return nbf.v4.new_code_cell(
        source,
        metadata={
            "cellView": "form",
            "jupyter": {"source_hidden": True},
            "tags": ["hide-input"],
        },
    )


def markdown_cells() -> list:
    return [
        nbf.v4.new_markdown_cell(
            r"""# On-device Online-SDFT: learn from the route you actually took

<a href="https://colab.research.google.com/github/lin826/SLM-Online-SDFT/blob/main/online_sdft_bandit_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

An on-device assistant must act before feedback exists. It can learn only after
the selected route produces a real OS/app callback. This notebook implements
that delayed mobile timeline with a **same-network** hindsight teacher and no
counterfactual outcomes.

This notebook makes that concrete with a real
[`LiquidAI/LFM2.5-230M`](https://huggingface.co/LiquidAI/LFM2.5-230M)
student. It is **self-contained**: the simulator and pipeline implementation
and the audited compact reference artifacts are embedded; it reads no
repository files. Strict reproduction uses the original Apple MPS/FP32 stack.
A T4 or CPU can run the same six-method protocol in portable mode, but different
numerical kernels are not claimed to produce identical bytes (Sections 5+).

### What you need

Sections 1–4 run in any Python notebook. Sections 5–6 need internet access and
Python 3.11+ to install a pinned LLM stack and download the public model. Every
notification and response is **synthetic**, and the run reproduces the
simulator rather than measuring latency or energy. Model use follows the
[`LFM1.0` license](https://huggingface.co/LiquidAI/LFM2.5-230M/blob/main/LICENSE).

### Path (play → understand → reproduce)

| Do this | Why |
| --- | --- |
| [1. Play the router](#1-play-the-router) | Feel the information bottleneck yourself |
| [2–4. Follow the pipeline](#2-protocol-at-a-glance) | See what is observed, delayed, and learned |
| [5. Reproduce](#5-reproduce-the-six-method-benchmark) | Run three paired seeds × all six methods and compare exact bytes |
| [6. Inspect](#6-plots-and-audits) | Plot every method and audit censored feedback |

### Map for RL / ML / AI engineers

| Term you already know | In this notebook |
| --- | --- |
| Contextual bandit $x_t$ | title/body + category/time/regime + local importance estimate |
| Bandit feedback | Observable user selection from the **executed** route, or `UNKNOWN` |
| Soft distillation / SDFT | Same-LFM hindsight + reliability-conditioned causal fusion |
| Hidden sampled preference | One seeded draw from $q_i\propto p_i^{100}$ after utility normalization ($T=0.01$); **simulation and accuracy only**, never training |
| Utility-optimal route | Full-information utility argmax; **regret only**, never training |
| Cumulative regret here | Σ utility gap vs clairvoyant $a^\star_t$, not best-in-class of $x$ alone |

> Long code cells are collapsed where the viewer supports it. Stay in the
> markdown + game unless you want implementation guts."""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Play the router

Use **only** the visible cues. Run the next two cells in order: initialize the
collapsed game engine once, then set `SCENARIO_ID` / `MY_ACTION` and run the
visible play cell. Commit mentally before you reveal the debrief.

| ID | Notification | Visible cues |
| ---: | --- | --- |
| `0` | “Mobile review starts in 10 minutes” | `calendar` · 15:19 · weekday · importance 0.97 |
| `1` | “Critical: Identity errors above threshold” | `monitoring` · 00:25 · on-call · importance 0.94 |
| `2` | “Maya sent you a message” | `social` · 20:11 · off-hours · importance 0.45 |
| `3` | “New grocery recommendations selected for you” | `promo` · 15:03 · weekday · importance 0.06 |

These are exact rows from seed 0 of the versioned
`semantic-title-body-sharp-t001` dataset. The engine cell is collapsed so its
hidden scoring state does not spoil the game."""
        ),
        reader_code_cell(GAME_ENGINE, "Hidden game engine and scenario state"),
        nbf.v4.new_code_cell(
            GAME_PLAY,
            metadata={"tags": ["game-input"], "jupyter": {"source_hidden": False}},
        ),
        nbf.v4.new_markdown_cell(
            """### What to notice

- A click does not prove the route maximized total utility (interruption cost).
- Only the chosen action creates an observable selection.
- A digest read under `LATER` reveals `LATER`, even when the hidden evaluator preferred an immediate open.
- Every `ARCHIVE` case stays `UNKNOWN`.
- The debrief is for you; the learner never sees the sampled preference or utility-optimal route.
- Feedback can improve the **next** decision only."""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. Protocol at a glance

The simulator's causal callback clock advances 15 minutes per decision; the
displayed local time is separate ordered context. An interrupt can expose an
immediate open after 1 minute, deletion after 15 minutes, or a
delayed read after 120 minutes. Digest observations mature after 120 minutes;
archive remains `UNKNOWN` after 240 minutes. A lesson can affect only requests
that arrive after its event-specific release. A matured `UNKNOWN` record is
audited and cannot become a route label or gradient update. The ICL/RAG
baselines may retain it only as explicitly unlabeled factual history.

Section 5 executes this exact delayed-feedback protocol for Base, ICL, RAG,
REINFORCE, RFT, and Online-SDFT on the three canonical paired streams."""
        ),
        nbf.v4.new_markdown_cell(
            """### REINFORCE benchmark configuration

The promoted REINFORCE baseline uses only matured callbacks from its executed
delivery surface. Its learner-only outcome map is `+5` for an immediate push
open, `-1` for a delayed push open, `-5` for a push deletion, `0` for a digest
open, and `-5` for a digest deletion. `UNKNOWN` is censored before mapping and
causes no update. This shaped training signal is separate from the shared
observable-reward metric reported for all methods.

It trains the common rank-4 Q/K/V/O LoRA adapter with learning rate `1e-4`,
batch size eight, a fixed zero baseline, entropy coefficient `1.0`, and
gradient-norm clipping at `1.0`. Those settings were selected in-sample on the
same canonical seeds 0–2, using exact action-match accuracy with utility regret
as a strict secondary gate; no disjoint confirmation seeds were run. All methods
retain the same `semantic-title-body-sharp-t001` streams and $T=0.01$ evaluator
sampling. The hidden preference, utilities, deadline, urgency, affinity, and
scenario taxonomy never enter the learner."""
        ),
        nbf.v4.new_markdown_cell(
            r"""## 3. Understand the setting

### Contextual-bandit contract

| Moment | Available | Sealed away |
| --- | --- | --- |
| Decision snapshot | title/body, category, local time, regime and importance | busy/interruption state, exact deadline/affinity, future callback, sampled preference, utility-optimal route |
| Student rollout | semantic context, parameters, matured past records | current callback, sampled preference, utility-optimal route |
| Observation window | user selection from the **selected** action, or `UNKNOWN` | both counterfactual outcomes |
| Future update | reliability-conditioned soft target: teacher + frozen decision prior + causal support; replay 64, batch 8 | scalar reward, sampled preference, or utility-optimal route as a label |

### Why this is not batch learning

| Batch | This online stream |
| --- | --- |
| Shuffled labeled set before serving | Each callback is consumed only after its delay expires |
| Shuffle + many epochs | One pass, in time order |
| Held-out exam afterward | Every live action is scored |
| Errors can vanish from test set | Cold-start errors stay in the metric |

The stream drifts weekday → on-call → off-hours. The objective is usefulness
**during** adaptation.

### Versioned evaluator sampling

The benchmark first normalizes evaluator utilities (shifting by the negative
minimum only when needed), then samples one hidden preference from
$q_i=p_i^{1/T}/\sum_j p_j^{1/T}$ at $T=0.01$, equivalently
$q_i\propto p_i^{100}$. The transform is evaluated in stable log space and
preserves exact zero support. Accuracy checks whether the executed action
exactly matches that sampled hidden preference. The utilities shape this
upstream distribution, but their numeric values are used directly only for
regret against the utility-optimal route; none of these evaluator-only fields
enters the learner.

The canonical seeds 0–2 have SHA-256 fingerprint
`986cdf1a7d5fcc04c2b33f1bf90a1fc4f24a97ee85e663370382d8a67e4c932d`.
Strict reproduction gates on that fingerprint, the audited source manifest,
the pinned model snapshot, and the exact three compact artifact byte strings."""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Follow the six-method pipeline

The notebook embeds the production stream, model wrapper, six method agents,
delayed-feedback loop, aggregation code, and compact serializer. Each seed gets
a fresh pinned LFM+LoRA instance; every method starts from the identical adapter
snapshot. Base, ICL, and RAG keep that zero-effect adapter frozen, while
REINFORCE, RFT, and Online-SDFT train the same 172,032-parameter LoRA capacity.

### Deployment contract

| Stage | Information |
| --- | --- |
| Student rollout | Visible `x_t` only |
| Decision snapshot | Title/body, category, local time, regime and importance |
| Teacher timing | After the selected route's callback matures |
| Teacher input | `x_t`, executed `a_t`, factual `f_t`, observed selection `s_t` or `UNKNOWN` |
| Initialization | The common zero-initialized rank-4 Q/K/V/O LoRA state |
| Retained target | Complete causal-fusion distribution: same-LFM teacher + frozen decision prior + causal support |
| Local update | A rank-4 Q/K/V/O LoRA adapter with 172,032 trainable parameters at learning rate `1e-3`; replay 64 with a 32-step recency half-life, batch 8, two steps after a four-example warm-up |
| Serving policy | 2% epsilon-greedy baseline; while max student confidence is at most 60%, mix a 15% `INTERRUPT` probe with an 80-step half-life; after step 160, taper the full exploratory distribution toward argmax with a five-step half-life |

The causal-fusion weights depend on callback reliability. Reliable singleton
callbacks use 5% fixed-initial teacher, 5% frozen decision prior, and 90%
causal support. For an ambiguous digest open, the frozen decision prior is
projected onto the compatible `{INTERRUPT, LATER}` support; the teacher gets
zero weight, so the learner preserves the uncertainty the callback leaves
unresolved. Replay sampling is recency-weighted and selection-balanced.
`ARCHIVE` cannot expose any preference, and `UNKNOWN` is audited but never
retained as a target. Serving uses the learned adapter only; no replay example
is inserted into the prompt."""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. Reproduce the six-method benchmark

This section executes 4,320 chronological decisions: three paired seeds × six
methods × 240 decisions. On the audited Apple MPS host it normally takes several
minutes; CPU can take substantially longer.

### 5.1 Install the LLM runtime

In Colab, **T4 GPU is preferred** for a portable run. A CPU-only runtime is
supported if GPU budget is exhausted. The setup cell
must run **before any cell that imports PyTorch**. It keeps Colab’s
CUDA-matched PyTorch and already-loaded numerical stack, pins the LLM package,
and verifies that the numerical and model runtime import cleanly. A normal
Colab run does not require a session restart.

Strict byte reproduction requires Python 3.11.9, NumPy 2.4.6, Torch 2.13.0,
Transformers 5.13.1, PEFT 0.19.1, Apple MPS, and FP32. Colab's CUDA/FP16 stack
can execute portable mode but cannot honestly claim byte identity with MPS.

> **Recovering after an earlier dependency or LFM model-loading error:** choose
> **Runtime → Disconnect and delete runtime**, reconnect, then run this corrected
> setup cell first. Those messages came from the old live dependency state,
> not from the embedded Online-SDFT pipeline."""
        ),
        reader_code_cell(INSTALL_DEPS, "Install the Liquid LFM runtime"),
        nbf.v4.new_markdown_cell(
            """### 5.2 Load the embedded simulator and pipeline

This long cell is copied from the production modules by the notebook builder.
It defines the stream, every baseline, delayed feedback, same-network teacher,
LoRA learners, aggregation, and exact compact serializer. It reads no external
repository file."""
        ),
        reader_code_cell(
            RUNTIME_GUARD + "\n\n" + embedded_core(),
            "Embedded production benchmark implementation",
        ),
        nbf.v4.new_markdown_cell(
            """### 5.3 Choose strict or portable execution

This cell keeps strict mode enabled on a local kernel and selects portable mode
automatically on Colab. Portable CUDA/CPU runs use the exact bundled canonical
events, but their final byte comparison is diagnostic rather than a release
claim because their numerical kernels differ from MPS."""
        ),
        nbf.v4.new_code_cell(
            RUN_CONFIGURATION,
            metadata={"tags": ["reproduction-config"]},
        ),
        nbf.v4.new_markdown_cell(
            """### 5.4 Run every seed and method

The pinned model snapshot is loaded fresh for each seed. Within a seed, all six
methods reuse one physical model and reset the common LoRA snapshot before acting."""
        ),
        reader_code_cell(RUNNER, "Run the canonical six-method benchmark in memory"),
        nbf.v4.new_markdown_cell(
            """### 5.5 Verify metrics and exact artifact bytes

The cell rebuilds `per_seed_metrics.csv`, `summary.json`, and
`qualitative_examples.json` with the embedded production serializer. It then
compares the generated bytes directly with the audited release bytes and shows
both SHA-256 digests. **Byte-identical reproduction** passes only when all three
files match; strict mode raises on the first artifact mismatch."""
        ),
        reader_code_cell(RESULTS, "Verify all six methods and exact result bytes"),
        nbf.v4.new_markdown_cell(
            """## 6. Plots and audits

Only useful after Section 5 has produced `metrics`, `curves`, and `rollouts` in
memory."""
        ),
        reader_code_cell(LEARNING_CURVES, "Plot all six learning trajectories"),
        reader_code_cell(AUDIT_INVARIANT, "Assert the ARCHIVE feedback invariant"),
        reader_code_cell(ACTION_MIX, "Plot executed action counts"),
        nbf.v4.new_markdown_cell(
            """## 7. Takeaway

Act with the semantic serving context. Wait for the selected route's observable
user selection, or `UNKNOWN`. Build a reliability-conditioned soft target from
same-network hindsight, the frozen decision-time prior, and causal support. Never use
either evaluator-only answer. Skip learning when the selection is censored, and queue
valid LoRA-adapter updates for future requests only.

Treat this as an executable simulator reproduction, not a phone-scale energy or
latency claim. Passing the byte gate proves that this notebook reproduced the
published simulator artifacts on the audited runtime; it is not production
deployment evidence.

When you close the notebook, the transferable idea is the causal loop, not the
particular notification categories. For the longer argument and figures, see the
[project website](https://lin826.github.io/SLM-Online-SDFT/)."""
        ),
    ]

def build_notebook():
    """Build a deterministic, clean notebook object without filesystem I/O."""
    notebook = nbf.v4.new_notebook(
        cells=markdown_cells(),
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        identity = f"{index}\0{cell.cell_type}\0{cell.source}".encode("utf-8")
        cell.id = hashlib.sha256(identity).hexdigest()[:16]
    return notebook


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook-only",
        action="store_true",
        help="regenerate only the notebook, leaving the GIF untouched",
    )
    args = parser.parse_args(argv)

    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK)
    print(f"wrote {NOTEBOOK}")
    if not args.notebook_only:
        GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
        namespace = {}
        exec(GIF_FUNCTIONS, namespace)
        gif_bytes = namespace["make_online_sdft_gif"]()
        GIF_PATH.write_bytes(gif_bytes)
        print(f"wrote {GIF_PATH}")


if __name__ == "__main__":
    main()

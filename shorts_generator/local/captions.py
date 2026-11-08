"""Burned-in captions and hook overlay as an ASS subtitle file.

Style: "word-highlight" — 2-4 words on screen at a time, the word being spoken
highlighted in colour, bold white text with a thick black outline, on one line
pinned at 70% of frame height (above the platform UI at the bottom). Lines never
overlap in time, so the caption never jumps. The hook overlay is a boxed title
of at most two balanced lines near the top for the first seconds.
"""
import os
import re
from typing import Dict, List, Optional

# Arial resolves to an installed bold sans almost everywhere (Liberation Sans on Linux).
CAPTION_FONT = os.getenv("CAPTION_FONT", "Arial")
CAPTION_HIGHLIGHT = os.getenv("CAPTION_HIGHLIGHT", "#FFD400")  # spoken-word colour
MAX_WORDS_PER_LINE = 3
MAX_CHARS_PER_LINE = 15      # one line at the caption size; captions never wrap
BREAK_PAUSE = 0.45          # start a new caption after a pause this long
HOOK_SECONDS = 3.0
CAPTION_Y = 0.70            # caption centre, as a fraction of frame height
HOOK_MAX_CHARS = 20          # fits one line at the hook size


def _ass_color(hex_rgb: str) -> str:
    """'#RRGGBB' -> ASS '&H00BBGGRR'."""
    h = hex_rgb.lstrip("#")
    if len(h) != 6:
        h = "FFD400"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def _ts(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean(word: str) -> str:
    w = word.strip()
    w = re.sub(r"[,;:]+$", "", w)          # trailing soft punctuation reads as clutter
    w = re.sub(r"(?<![.])\.$", "", w)      # drop a single trailing period, keep "..."
    return w.replace("{", "(").replace("}", ")").replace("\\", "/")


def group_words(words: List[Dict]) -> List[List[Dict]]:
    """Split words into short caption lines, breaking on pauses and sentence ends."""
    joined: List[Dict] = []
    for w in words:
        # Whisper splits "U.S." into "u" + ".s" and "10,000" into "10" + ",000"; glue them back.
        if joined and re.match(r"[.,]\w", w["word"].strip()):
            joined[-1] = {**joined[-1], "word": joined[-1]["word"].strip() + w["word"].strip(), "end": w["end"]}
        else:
            joined.append(w)
    groups: List[List[Dict]] = []
    cur: List[Dict] = []
    for w in joined:
        text = _clean(w["word"])
        if not text:
            continue
        w = {**w, "text": text}
        if cur:
            gap = w["start"] - cur[-1]["end"]
            chars = sum(len(x["text"]) + 1 for x in cur) + len(text)
            ended = bool(re.search(r"[.!?…][\"')\]]*$", cur[-1]["word"].strip()))
            if len(cur) >= MAX_WORDS_PER_LINE or chars > MAX_CHARS_PER_LINE or gap >= BREAK_PAUSE or ended:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def _wrap(text: str, width: int = HOOK_MAX_CHARS) -> str:
    """One line if it fits, else the most even two-line split, else greedy lines (at most 3)."""
    words = text.split()
    if len(text) <= width:
        return " ".join(words)
    splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    fitting = [s for s in splits if max(len(s[0]), len(s[1])) <= width]
    if fitting:
        return r"\N".join(min(fitting, key=lambda s: (max(len(s[0]), len(s[1])), len(s[0]))))
    out, line = [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return r"\N".join(out[:3])


def build_ass(
    words: List[Dict],
    clip_duration: float,
    width: int = 1080,
    height: int = 1920,
    hook_text: Optional[str] = None,
    captions: bool = True,
) -> str:
    """Return the full .ass document. `words` have clip-relative start/end."""
    font_size = int(height * 0.052)          # ~100 px at 1920: readable on a phone, 15 chars fit one line
    hook_size = int(height * 0.044)
    hi = _ass_color(CAPTION_HIGHLIGHT)
    # A fixed point instead of a margin: lines never shift, and libass never stacks two of them.
    pos = f"{{\\an5\\pos({width // 2},{int(height * CAPTION_Y)})}}"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{CAPTION_FONT},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{max(4, font_size // 12)},2,5,60,60,0,1
Style: Hook,{CAPTION_FONT},{hook_size},&H00101010,&H00101010,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,4,{max(10, hook_size // 4)},0,8,80,80,{int(height * 0.09)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: List[str] = []

    if hook_text and hook_text.strip():
        text = _wrap(_clean(hook_text.strip()).replace("\n", " "))
        end = min(HOOK_SECONDS, max(1.0, clip_duration))
        events.append(f"Dialogue: 1,{_ts(0)},{_ts(end)},Hook,,0,0,0,,{{\\fad(120,250)}}{text}")

    if captions:
        caps = []
        groups = group_words(words)
        for gi, g in enumerate(groups):
            g_end = g[-1]["end"]
            if gi + 1 < len(groups) and groups[gi + 1][0]["start"] - g_end < 0.35:
                g_end = groups[gi + 1][0]["start"]   # no flicker between close lines
            g_end = min(g_end + 0.05, clip_duration)
            for k, w in enumerate(g):
                start = g[0]["start"] if k == 0 else w["start"]
                end = g[k + 1]["start"] if k + 1 < len(g) else g_end
                if end <= start:
                    continue
                parts = []
                for m, x in enumerate(g):
                    if m == k:
                        parts.append(f"{{\\c{hi}}}{x['text']}{{\\c&H00FFFFFF&}}")
                    else:
                        parts.append(x["text"])
                caps.append([start, end, " ".join(parts)])
        caps.sort(key=lambda c: c[0])
        for i, (start, end, text) in enumerate(caps):
            if i + 1 < len(caps):
                end = min(end, caps[i + 1][0])  # Whisper word times overlap; two lines on screen would stack
            if end > start:
                events.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Caption,,0,0,0,,{pos}{text}")

    return header + "\n".join(events) + "\n"

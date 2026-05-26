"""
Semantic Intent Parser for SceneShift — Layer 1 of 3.

Responsibility:
    Extract WHAT the user wants from a natural language prompt.
    This module performs NLP understanding ONLY.

    It does NOT decide:
    - which models to activate
    - whether Stable Diffusion is needed
    - pipeline routing or stage activation

    All execution decisions are delegated to execution_planner.py (Layer 2).

Public API:
    parse_prompt(prompt: str) -> ParsedIntent
    find_solid_color(text: str) -> Optional[str]
    parse_command(command: str) -> Tuple[List[str], Optional[str], bool]  # legacy shim
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ── Color lookup table ─────────────────────────────────────────────────────────

COLOR_MAP: Dict[str, str] = {
    "white":           "#FFFFFF",
    "black":           "#000000",
    "blue":            "#0000FF",
    "red":             "#FF0000",
    "green":           "#00FF00",
    "gray":            "#808080",
    "grey":            "#808080",
    "yellow":          "#FFFF00",
    "cyan":            "#00FFFF",
    "magenta":         "#FF00FF",
    "orange":          "#FFA500",
    "purple":          "#800080",
    "pink":            "#FFC0CB",
    "brown":           "#A52A2A",
    "navy":            "#000080",
    "dark navy":       "#001F3F",
    "sky blue":        "#87CEEB",
    "light sky blue":  "#87CEEB",
    "light blue":      "#ADD8E6",
    "dark blue":       "#00008B",
    "gold":            "#FFD700",
    "silver":          "#C0C0C0",
    "beige":           "#F5F5DC",
    "ivory":           "#FFFFF0",
    "teal":            "#008080",
    "maroon":          "#800000",
    "olive":           "#808000",
    "coral":           "#FF7F50",
    "lavender":        "#E6E6FA",
    "turquoise":       "#40E0D0",
    "indigo":          "#4B0082",
    "violet":          "#EE82EE",
    "crimson":         "#DC143C",
    "lime":            "#00FF00",
    "khaki":           "#F0E68C",
    "salmon":          "#FA8072",
    "tan":             "#D2B48C",
    "chocolate":       "#D2691E",
    "tomato":          "#FF6347",
    "wheat":           "#F5DEB3",
    "mint":            "#98FF98",
}

# ── Style vocabulary ───────────────────────────────────────────────────────────

# Ordered longest-first so multi-word phrases match before single words
STYLE_VOCABULARY: List[str] = sorted([
    "cinematic look", "cinematic style", "cinematic",
    "film noir", "film grain", "film",
    "passport style", "passport photo", "passport",
    "id photo", "id card style",
    "professional headshot", "professional",
    "corporate",
    "studio lighting", "studio look", "studio",
    "photorealistic", "photo realistic", "realistic",
    "sci-fi", "science fiction",
    "fantasy world", "fantasy",
    "cyberpunk neon", "cyberpunk",
    "anime style", "anime",
    "vintage look", "vintage",
    "retro style", "retro",
    "bokeh background", "bokeh",
    "shallow depth of field", "depth of field",
    "blurred background", "blur background",
    "business casual", "business",
    "editorial", "commercial",
    "lifestyle",
    "neon lights", "neon",
    "futuristic",
    "minimalist", "clean look",
], key=len, reverse=True)

# ── Intent signals ─────────────────────────────────────────────────────────────

# Verbs that signal object removal
_REMOVAL_VERBS = (
    r"remove(?:\s+the)?",
    r"erase(?:\s+the)?",
    r"delete(?:\s+the)?",
    r"clean(?:\s+up)?(?:\s+the)?",
    r"eliminate(?:\s+the)?",
    r"get\s+rid\s+of(?:\s+the)?",
    r"take\s+out(?:\s+the)?",
    r"wipe\s+out(?:\s+the)?",
    r"cut\s+out(?:\s+the)?",
    r"strip(?:\s+the)?",
    r"hide(?:\s+the)?",
    r"mask(?:\s+out)?(?:\s+the)?",
)
_REMOVAL_VERB_PATTERN = r"(?:" + "|".join(_REMOVAL_VERBS) + r")"

# Words that are background references, not objects to remove
_BG_STOPWORDS = {
    "background", "bg", "backdrop", "scene", "environment",
    "everything", "all", "nothing",
}

# Determiners/adjectives to strip from extracted object names
_ARTICLE_PATTERN = re.compile(
    r"^\s*(?:a|an|the|some|any|all|that|this|those|these|my|"
    r"unwanted|hanging|extra|visible|stray|left|right|front|"
    r"upper|lower|old|new|small|large|big|little)\s+",
    re.IGNORECASE,
)

# Keywords signalling foreground/identity preservation
_IDENTITY_SIGNALS = frozenset([
    "keep my face", "don't change my face", "preserve my face",
    "protect face", "preserve face", "keep face",
    "passport", "id photo", "id card style",
    "headshot", "portrait", "keep identity", "preserve identity",
    "don't alter me", "don't change me",
])

_FOREGROUND_PRESERVE_SIGNALS = frozenset([
    "only the background", "only background", "just the background",
    "just background", "keep the subject", "keep subject",
    "keep me", "keep person", "don't touch me", "don't change me",
    "preserve subject", "preserve the subject", "foreground only",
    "background only",
])

# Transparent-background signals
_TRANSPARENT_SIGNALS = frozenset([
    "transparent background", "transparent bg", "transparency",
    "remove background", "remove the background", "remove bg",
    "delete background", "delete bg", "clear background", "clear bg",
    "no background", "no bg", "see through background",
    "cutout", "cut out background", "png background",
])


# ── BackgroundType enum ───────────────────────────────────────────────────────

class BackgroundType(Enum):
    """
    Explicit semantic classification of the background request.

    Determined entirely at parse time — zero model activation logic here.
    The execution_planner maps each type to a BackgroundStrategy.

    Priority order (highest to lowest):
        CUSTOM_IMAGE    — user uploaded a file (set by planner, not parser)
        TRANSPARENT     — alpha cutout, zero generation
        SOLID_COLOR     — flat hex/rgb fill, zero generation
        STUDIO          — neutral/plain/professional, defaults to white fill
        GENERATED_SCENE — real-world or fantastical environment, requires SD
    """
    NONE            = "none"
    SOLID_COLOR     = "solid_color"      # Named or hex/rgb color — flat fill
    TRANSPARENT     = "transparent"      # Alpha channel cutout
    STUDIO          = "studio"           # Plain / neutral / professional
    GENERATED_SCENE = "generated_scene"  # Office, beach, city, etc.
    CUSTOM_IMAGE    = "custom_image"     # User-uploaded image (set by planner)


# Keywords that classify an extracted background description as STUDIO (not generative)
_STUDIO_BG_KEYWORDS: frozenset = frozenset([
    "plain", "flat", "clean", "neutral", "solid",
    "studio", "simple", "minimal", "basic", "standard",
    "blank", "empty", "uniform", "matte", "single color",
    "single colour", "monotone", "monochrome", "one color", "one colour",
])

# Keywords that classify an extracted background description as GENERATED_SCENE
_SCENE_BG_KEYWORDS: frozenset = frozenset([
    "office", "workspace", "meeting room", "boardroom", "conference room",
    "beach", "ocean", "sea", "coast", "seaside",
    "forest", "jungle", "woods", "trees",
    "city", "urban", "street", "downtown", "skyline", "cityscape", "metropolis",
    "mountain", "hills", "valley", "canyon", "cliff",
    "space", "galaxy", "stars", "cosmos", "universe", "nebula",
    "kitchen", "dining room", "living room", "bedroom", "bathroom",
    "library", "bookshelf", "study room",
    "park", "garden", "field", "meadow", "lawn",
    "restaurant", "cafe", "coffee shop", "bar",
    "gym", "sports", "stadium",
    "school", "classroom", "university", "campus",
    "airport", "train station", "station",
    "sunset", "sunrise", "golden hour", "dusk", "dawn",
    "night", "evening", "morning",
    "rainy", "snowy", "foggy", "misty",
    "underwater", "cave", "ruins", "temple",
    "nature", "outdoor", "outside",
    "indoor", "inside", "interior", "room",
    "landscape", "scenery", "environment",
    "fantasy", "sci-fi", "cyberpunk", "futuristic",
    "anime", "cartoon",
    "vintage", "retro",
    "neon lights", "neon",
])


def _classify_background_type(raw_bg: str) -> BackgroundType:
    """
    Classify a raw extracted background description into a BackgroundType.

    Classification priority:
        1. Strict TRANSPARENT check
        2. Strict SOLID_COLOR check
        3. Strict GENERATED_SCENE check
        4. Fallbacks (named colors, scene/studio keyword dictionaries)
    """
    t = raw_bg.lower()

    # 1. Strict TRANSPARENT check
    transparent_kws = ["transparent background", "remove background", "png", "clear background"]
    if any(kw in t for kw in transparent_kws):
        return BackgroundType.TRANSPARENT

    # 2. Strict SOLID_COLOR check
    solid_kws = [
        "blue background", "green background", "white background", "black background",
        "plain background", "passport background", "studio background", "solid color"
    ]
    if any(kw in t for kw in solid_kws):
        return BackgroundType.SOLID_COLOR

    # 3. Strict GENERATED_SCENE check
    scene_kws = [
        "beach", "mountain", "office", "cyberpunk city",
        "fantasy world", "sci-fi environment", "cinematic landscape"
    ]
    if any(kw in t for kw in scene_kws):
        return BackgroundType.GENERATED_SCENE

    # 4. Fallback heuristics for broader keyword dictionaries & named colors
    if any(kw in t for kw in _SCENE_BG_KEYWORDS):
        if "studio" in t:
            generic_scenes = {"room", "interior", "indoor", "inside"}
            non_generic_scenes = _SCENE_BG_KEYWORDS - generic_scenes
            if not any(kw in t for kw in non_generic_scenes):
                return BackgroundType.STUDIO
        return BackgroundType.GENERATED_SCENE

    if find_solid_color(raw_bg) is not None:
        return BackgroundType.SOLID_COLOR

    if any(kw in t for kw in _STUDIO_BG_KEYWORDS):
        return BackgroundType.STUDIO

    return BackgroundType.STUDIO


# ── ParsedIntent dataclass ─────────────────────────────────────────────────────

@dataclass
class ParsedIntent:
    """
    Pure semantic extraction of user intent.

    Contains NO model activation flags and NO pipeline routing decisions.
    Every field answers "WHAT does the user want?" — not "HOW to do it?".

    Execution routing is the sole responsibility of execution_planner.py.
    """

    # Object removal — any objects the user asked to erase
    remove_targets: List[str] = field(default_factory=list)

    # Object replacement — {source_object: replacement_description}
    replace_targets: Dict[str, str] = field(default_factory=dict)

    # Background — either a free-text scene description OR a resolved solid color
    background_request: Optional[str] = None     # e.g. "busy city office", "beach at sunset"
    background_color: Optional[str] = None       # Hex "#RRGGBB" / "rgb(r,g,b)" / "transparent"

    # Explicit background classification — set by parser, used by planner for routing
    background_type: Optional[BackgroundType] = None

    # Style descriptors found in the prompt (raw strings, not canonical names)
    style_descriptors: List[str] = field(default_factory=list)

    # Semantic preservation flags (inferred from wording, not model decisions)
    preserve_identity: bool = False          # User implies face/person must not change
    preserve_foreground: bool = False        # User wants ONLY background changed

    # Explicit transparency request
    transparency_requested: bool = False

    # Diagnostics
    raw_prompt: str = ""
    parse_confidence: float = 0.0


# ── Color utility ──────────────────────────────────────────────────────────────

def find_solid_color(text: str) -> Optional[str]:
    """
    Attempt to resolve a solid color from text.

    Priority:
        1. Hex color literal (#RRGGBB / #RGB)
        2. RGB function notation rgb(r,g,b)
        3. Named color from COLOR_MAP
        4. Implicit solid keywords (solid, plain, studio, passport) → white

    Returns:
        Color string ("#RRGGBB" / "rgb(r,g,b)") or None if no solid color found.
    """
    t = text.lower().strip()

    # 1. Hex
    hex_match = re.search(r"#([0-9a-fA-F]{3,6})\b", t)
    if hex_match:
        val = hex_match.group(1)
        if len(val) == 3:
            val = "".join(c * 2 for c in val)
        return f"#{val.upper()}"

    # 2. RGB function
    rgb_match = re.search(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", t)
    if rgb_match:
        r, g, b = map(int, rgb_match.groups())
        return f"rgb({r},{g},{b})"

    # 3. Named colors (longest-match first to catch "dark navy" before "navy")
    for name in sorted(COLOR_MAP, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", t):
            return COLOR_MAP[name]

    # 4. Implicit solid keywords → default white
    _IMPLICIT_SOLID = (
        "solid color", "plain color", "studio color",
        "solid bg", "solid background", "plain bg", "plain background",
        "studio bg", "studio background",
        "passport", "solid", "plain",
    )
    if any(k in t for k in _IMPLICIT_SOLID):
        return "#FFFFFF"

    return None


# ── Internal extraction helpers ────────────────────────────────────────────────

def _extract_removal_targets(text: str) -> List[str]:
    """
    Multi-pass extraction of objects the user wants removed.

    Pass 1: dependency-like regex (verb + noun-phrase)
    Pass 2: list splitting on comma / 'and'
    Pass 3: article/adjective stripping + stopword filtering
    """
    targets: List[str] = []
    seen: set = set()

    # Pattern: <removal_verb> <optional_det> <object_phrase>
    # Stop at: " and " followed by another verb, background action, or sentence end
    stop_clause = (
        r"(?=\s+(?:and\s+)?(?:change|replace|make|set|use|style|with|to|so|then|also|"
        r"remove|erase|delete|clean|eliminate|get|take|wipe|cut|strip|hide|mask)\b)"
    )
    pattern = re.compile(
        _REMOVAL_VERB_PATTERN
        + r"\s+"
        + r"((?:.+?)(?:" + stop_clause + r"|$))",
        re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        # Safety check: if the removal verb is preceded by an article, preposition,
        # or active background verb, it is likely an adjective/noun usage (e.g. "a clean white room")
        # rather than an action verb.
        prefix = text[:match.start()].rstrip(" ,;.-")
        if re.search(r"\b(to|with|as|a|an|the|bg|background|backdrop|scene|make|change|set|replace)\s*$", prefix, re.IGNORECASE):
            continue

        raw_phrase = match.group(1).strip()
        # Split on comma or " and "
        parts = re.split(r",\s*|\s+and\s+", raw_phrase)
        for part in parts:
            obj = _clean_object_name(part)
            if obj and obj not in seen:
                targets.append(obj)
                seen.add(obj)

    return targets


def _clean_object_name(raw: str) -> str:
    """Strip articles, normalize whitespace, filter stopwords."""
    cleaned = _ARTICLE_PATTERN.sub("", raw).strip().lower()
    # Remove trailing punctuation
    cleaned = re.sub(r"[^\w\s\-]$", "", cleaned).strip()
    if cleaned in _BG_STOPWORDS or not cleaned:
        return ""
    return cleaned


def _extract_replace_targets(text: str) -> Dict[str, str]:
    """
    Extract object replacement instructions.
    Pattern: "replace <source> with <replacement>"
    """
    replacements: Dict[str, str] = {}
    pattern = re.compile(
        r"\breplace\b\s+([\w\s\-]+?)\s+\bwith\b\s+([\w\s\-,]+?)(?:\s+and\b|$)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        source = _clean_object_name(match.group(1))
        replacement = match.group(2).strip().lower()
        # Exclude background-level replacements (handled by BG extraction)
        if source and source not in _BG_STOPWORDS:
            replacements[source] = replacement
    return replacements


def _extract_background(text: str) -> Tuple[Optional[str], Optional[str], Optional[BackgroundType]]:
    """
    Extract background intent and classify it into a BackgroundType.

    Returns:
        (background_request, background_color, background_type)

        background_request: free-text scene description (for GENERATED_SCENE)
        background_color:   hex/rgb string or "transparent" (for SOLID_COLOR / TRANSPARENT)
        background_type:    BackgroundType classification used by execution_planner

        All three can be None if no background instruction is found.

    Regex pattern groups (tried in order, first match wins):
        P1  — action verb + background + connector + description
              e.g. "change background to blue", "set the bg to office"
        P2  — background + connector + description
              e.g. "background to blue", "bg with beach sunset"
        P3  — background + modal/linking verb + description
              e.g. "background should be green", "bg must be white", "bg is red"
        P4  — background + equals/dash separator + description
              e.g. "background = white", "background - blue", "background: navy"
        P5  — background + colour/color keyword + description
              e.g. "background colour blue", "bg color red"
        P6  — bg/background keyword ONLY followed directly by color/description
              e.g. "bg blue", "bg red", "bg office"
        P7  — description BEFORE background keyword (adjective-noun order)
              e.g. "white background", "solid green background", "cyberpunk background"
    """
    t = text.lower().strip()

    bg_match = None
    matched_group = 1  # Which capture group holds the description

    # Stop background extraction if we run into removal or preservation instructions
    stop_clause = (
        r"(?=\s+(?:and\s+)?(?:remove|erase|delete|clean|eliminate|get|take|wipe|cut|strip|hide|mask|keep|preserve|protect)\b)"
    )

    # ── P1: action verb + background + optional connector ────────────────────────
    # Matches: "change background to blue", "make bg a beach", "set the bg to office"
    if not bg_match:
        bg_match = re.search(
            r"(?:change|replace|make|set|use|put|add|apply|switch(?:\s+to)?)\s+"
            r"(?:the\s+)?(?:background|bg|backdrop|scene)\s+"
            r"(?:with|to|be|as|into|for)?\s*"
            r"((?:.+?)(?:" + stop_clause + r"|$))",
            t, re.IGNORECASE,
        )

    # ── P2: background + connector word + description ────────────────────────────
    # Matches: "background to blue", "bg with beach", "backdrop as white"
    if not bg_match:
        bg_match = re.search(
            r"(?:background|bg|backdrop)\s+(?:to|with|be|as|into)\s+"
            r"((?:.+?)(?:" + stop_clause + r"|$))",
            t, re.IGNORECASE,
        )

    # ── P3: background + modal/linking verb + description ────────────────────────
    # Matches: "background should be green", "bg must be white", "background is red",
    #          "background needs to be blue", "background has to be navy"
    if not bg_match:
        bg_match = re.search(
            r"(?:background|bg|backdrop)\s+"
            r"(?:should(?:\s+be)?|must(?:\s+be)?|needs?\s+to\s+be|has\s+to\s+be|"
            r"will(?:\s+be)?|is|are|was|would\s+be|could\s+be)\s+"
            r"(?:a\s+|an\s+|the\s+)?((?:.+?)(?:" + stop_clause + r"|$))",
            t, re.IGNORECASE,
        )

    # ── P4: background + separator (=, -, —, |) + description ────────────────────
    # Matches: "background = white", "background - blue", "background: navy"
    if not bg_match:
        bg_match = re.search(
            r"(?:background|bg|backdrop)\s*[=:\-\u2014\u2013]\s*"
            r"((?:.+?)(?:" + stop_clause + r"|$))",
            t, re.IGNORECASE,
        )

    # ── P5: background + colour/color keyword + description ──────────────────────
    # Matches: "background colour blue", "bg color red", "background colour: navy"
    if not bg_match:
        bg_match = re.search(
            r"(?:background|bg)\s+colou?r\s*:?\s*"
            r"((?:.+?)(?:" + stop_clause + r"|$))",
            t, re.IGNORECASE,
        )

    # ── P6: bg abbreviation + direct description (no connector) ──────────────────
    # Matches: "bg blue", "bg red", "bg office", "bg beach"
    # Requires \b on both sides to avoid matching mid-word
    if not bg_match:
        bg_match = re.search(
            r"\bbg\b\s+([a-z][a-z0-9 \-]+?)(?:\s+(?:please|now|for me))?$",
            t, re.IGNORECASE,
        )

    # ── P7: description BEFORE background keyword (adjective-noun) ───────────────
    # Matches: "white background", "solid green background", "beach background"
    if not bg_match:
        bg_match = re.search(
            r"((?:.+?)(?:" + stop_clause + r"|(?=\s+(?:background|bg|backdrop)\b)))\s+(?:background|bg|backdrop)\b",
            t, re.IGNORECASE,
        )

    if bg_match:
        raw_bg = bg_match.group(matched_group).strip()

        # Strip leading articles/prepositions that may be captured
        raw_bg = re.sub(
            r"^(?:with|to|be|as|a|an|the|into|for|about)\s+",
            "", raw_bg,
        ).strip()

        if not raw_bg:
            return None, None, None

        # ── Transparency check (before color — "transparent" has no color) ───────
        if any(w in raw_bg for w in ("transparent", "transparency", "none", "cutout", "png")):
            return None, "transparent", BackgroundType.TRANSPARENT

        # ── Classify: scene overrides color ("blue office" = GENERATED_SCENE) ────
        bg_type = _classify_background_type(raw_bg)

        if bg_type == BackgroundType.SOLID_COLOR:
            color_val = find_solid_color(raw_bg)
            return None, color_val, BackgroundType.SOLID_COLOR

        if bg_type == BackgroundType.STUDIO:
            # Studio implies a solid white fill — resolve any named color if present
            color_val = find_solid_color(raw_bg) or "#FFFFFF"
            return None, color_val, BackgroundType.STUDIO

        # GENERATED_SCENE — return the raw description for SD prompt building
        return raw_bg, None, BackgroundType.GENERATED_SCENE

    return None, None, None


def _extract_style_descriptors(text: str) -> List[str]:
    """
    Scan the full prompt for known style vocabulary.
    Returns all matched descriptors (raw strings), longest-match first.
    """
    t = text.lower()
    found: List[str] = []
    covered: set = set()

    for phrase in STYLE_VOCABULARY:
        if phrase in t:
            # Ensure we don't double-count sub-phrases already covered
            start = t.find(phrase)
            span = (start, start + len(phrase))
            if not any(s <= span[0] and span[1] <= e for (s, e) in covered):
                found.append(phrase)
                covered.add(span)

    return found


def _detect_preserve_identity(text: str) -> bool:
    t = text.lower()
    return any(signal in t for signal in _IDENTITY_SIGNALS)


def _detect_preserve_foreground(text: str) -> bool:
    t = text.lower()
    return any(signal in t for signal in _FOREGROUND_PRESERVE_SIGNALS)


def _detect_transparency(text: str) -> bool:
    t = text.lower()
    return any(signal in t for signal in _TRANSPARENT_SIGNALS)


def _compute_confidence(
    prompt: str,
    remove_targets: List[str],
    background_request: Optional[str],
    background_color: Optional[str],
    style_descriptors: List[str],
    transparency_requested: bool,
) -> float:
    """
    Heuristic confidence: fraction of prompt tokens that were semantically matched.
    """
    tokens = re.findall(r"\b\w+\b", prompt.lower())
    if not tokens:
        return 0.0

    matched_tokens: set = set()

    for obj in remove_targets:
        matched_tokens.update(obj.split())
    if background_request:
        matched_tokens.update(background_request.split())
    if background_color:
        # A color was found — count the color name tokens
        for name in COLOR_MAP:
            if name in prompt.lower():
                matched_tokens.update(name.split())
    for style in style_descriptors:
        matched_tokens.update(style.split())
    if transparency_requested:
        matched_tokens.update(["transparent", "background"])

    matched_count = sum(1 for t in tokens if t in matched_tokens)
    raw = matched_count / len(tokens)
    return round(min(1.0, raw * 1.3), 3)   # slight boost for partial matches


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_prompt(prompt: str) -> ParsedIntent:
    """
    Parse a natural language editing prompt into a structured ParsedIntent.

    This is a pure NLP operation — no model activation decisions are made here.
    Pass the returned ParsedIntent to execution_planner.build_execution_plan()
    to obtain an ExecutionPlan with concrete model flags.

    Args:
        prompt: Raw user instruction string.

    Returns:
        ParsedIntent with all semantic fields populated.

    Examples:
        >>> intent = parse_prompt("remove tissues and change background to blue")
        >>> intent.remove_targets
        ['tissues']
        >>> intent.background_color
        '#0000FF'

        >>> intent = parse_prompt("make passport photo")
        >>> intent.preserve_identity
        True
        >>> intent.background_color
        '#FFFFFF'

        >>> intent = parse_prompt("replace background with busy office")
        >>> intent.background_request
        'busy office'
    """
    if not prompt or not prompt.strip():
        return ParsedIntent(raw_prompt=prompt or "")

    text = prompt.strip()

    # ── Pass 1: Transparency (check before other BG extraction) ───────────────
    transparency_requested = _detect_transparency(text)

    # ── Pass 2: Object removal ────────────────────────────────────────────────
    remove_targets = _extract_removal_targets(text)

    # ── Pass 3: Object replacement ────────────────────────────────────────────
    replace_targets = _extract_replace_targets(text)

    # ── Pass 4: Background ────────────────────────────────────────────────────
    if transparency_requested:
        background_request = None
        background_color = "transparent"
        background_type: Optional[BackgroundType] = BackgroundType.TRANSPARENT
    else:
        background_request, background_color, background_type = _extract_background(text)

    # ── Pass 5: Style descriptors ─────────────────────────────────────────────
    style_descriptors = _extract_style_descriptors(text)

    # ── Pass 6: Preservation flags ────────────────────────────────────────────
    preserve_identity = _detect_preserve_identity(text)
    preserve_foreground = _detect_preserve_foreground(text)

    # Passport/id photo implies identity preservation
    if any(s in ("passport", "passport photo", "passport style", "id photo") for s in style_descriptors):
        preserve_identity = True

    # ── Pass 7: Confidence ────────────────────────────────────────────────────
    confidence = _compute_confidence(
        text,
        remove_targets,
        background_request,
        background_color,
        style_descriptors,
        transparency_requested,
    )

    return ParsedIntent(
        remove_targets=remove_targets,
        replace_targets=replace_targets,
        background_request=background_request,
        background_color=background_color,
        background_type=background_type,
        style_descriptors=style_descriptors,
        preserve_identity=preserve_identity,
        preserve_foreground=preserve_foreground,
        transparency_requested=transparency_requested,
        raw_prompt=text,
        parse_confidence=confidence,
    )


# ── Backwards-compatibility shim ───────────────────────────────────────────────

def parse_command(command: str) -> Tuple[List[str], Optional[str], bool]:
    """
    Deprecated compatibility shim.

    Calls parse_prompt() internally and converts the result back to the
    legacy 3-tuple format used by orchestrator.py prior to the 3-layer refactor.

    Prefer parse_prompt() + execution_planner.build_execution_plan() for new code.
    """
    if not command:
        return [], None, False

    intent = parse_prompt(command)

    # Map background_color → legacy background_prompt format
    if intent.transparency_requested or intent.background_color == "transparent":
        background_prompt: Optional[str] = "transparent"
        solid_background = True
    elif intent.background_color:
        # Convert raw hex/rgb to legacy "color:#RRGGBB" format
        color = intent.background_color
        if color.startswith("#") or color.startswith("rgb"):
            background_prompt = f"color:{color}"
        else:
            background_prompt = color
        solid_background = True
    elif intent.background_request:
        background_prompt = intent.background_request
        solid_background = False
    else:
        background_prompt = None
        solid_background = False

    return intent.remove_targets, background_prompt, solid_background


def parse_raw_background_prompt(bg_prompt: str) -> Tuple[Optional[str], Optional[str], BackgroundType]:
    """
    Parse and classify an isolated background prompt string (e.g., from direct API calls).

    Returns:
        Tuple of (background_request, background_color, background_type)
    """
    t = bg_prompt.lower().strip()
    if not t:
        return None, None, BackgroundType.STUDIO

    # 1. Transparency check
    if any(w in t for w in ("transparent", "transparency", "none", "cutout", "png")):
        return None, "transparent", BackgroundType.TRANSPARENT

    # 2. Classify background type
    bg_type = _classify_background_type(t)

    # 3. Resolve solid/studio color values
    if bg_type == BackgroundType.SOLID_COLOR:
        color_val = find_solid_color(t)
        return None, color_val, BackgroundType.SOLID_COLOR

    if bg_type == BackgroundType.STUDIO:
        color_val = find_solid_color(t) or "#FFFFFF"
        return None, color_val, BackgroundType.STUDIO

    # 4. Generated scene
    return bg_prompt, None, BackgroundType.GENERATED_SCENE

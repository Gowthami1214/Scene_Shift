"""
Execution Planner for SceneShift — Layer 2 of 3.

Responsibility:
    Convert a ParsedIntent (Layer 1) into a deterministic ExecutionPlan
    that the Pipeline Orchestrator (Layer 3) follows without further inference.

Key safety principle:
    Stable Diffusion (both text-to-image and inpainting) must be EXPLICITLY
    approved by this planner. It is NEVER activated by default.
    This prevents expensive or destructive generation for simple requests
    like "white background" or "remove tissues".

Architecture:
    parse_prompt()          → ParsedIntent   (what the user wants)
    build_execution_plan()  → ExecutionPlan  (how to achieve it safely)
    orchestrator.run()      → PipelineResult (execute the plan)

Public API:
    build_execution_plan(intent, ...) -> ExecutionPlan
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.pipeline.prompt_parser import ParsedIntent, BackgroundType


# ── Strategy Enums ─────────────────────────────────────────────────────────────

class BackgroundStrategy(str, Enum):
    """Determines how the background is produced. Ordered by compute cost (cheapest first)."""
    NONE             = "none"             # No background change requested
    TRANSPARENT_FILL = "transparent_fill" # Alpha extraction only — zero generation cost
    COLOR_FILL       = "color_fill"       # NumPy solid fill — zero generation cost
    CUSTOM_IMAGE     = "custom_image"     # User-uploaded image — no generation needed
    PROCEDURAL       = "procedural"       # Gradient/keyword-driven OpenCV — no SD
    SD_GENERATION    = "sd_generation"    # Stable Diffusion text-to-image — expensive


class ForegroundStrategy(str, Enum):
    """Determines how the subject/foreground is processed."""
    PRESERVE_PIXELS  = "preserve_pixels"  # Pass through unchanged — zero cost
    LAMA_INPAINTING  = "lama_inpainting"  # LaMa localized fill — moderate cost
    SD_INPAINTING    = "sd_inpainting"    # Stable Diffusion inpainting — expensive


class SegmentationStrategy(str, Enum):
    """Determines which segmentation model runs (if any)."""
    SKIP             = "skip"             # Segmentation not needed
    AUTO_YOLO        = "auto_yolo"        # YOLO automatic foreground detection
    INTERACTIVE_SAM  = "interactive_sam"  # SAM2 with user click points


# ── ExecutionPlan dataclass ────────────────────────────────────────────────────

@dataclass
class ExecutionPlan:
    """
    Deterministic, model-level execution plan.

    The orchestrator reads this plan directly. It performs no further inference
    or conditional logic — it simply executes what the plan specifies.

    SD safety guards:
        sd_generation_approved  — must be True before BackgroundGenerator(SD) is called
        sd_inpainting_approved  — must be True before ObjectEditor(SD) is called
        Both default to False and must be explicitly set True by the planner.
    """

    # ── Object removal ─────────────────────────────────────────────────────────
    run_object_removal: bool = False
    removal_targets: List[str] = field(default_factory=list)

    # ── Foreground handling ────────────────────────────────────────────────────
    foreground_strategy: ForegroundStrategy = ForegroundStrategy.PRESERVE_PIXELS

    # ── Segmentation ──────────────────────────────────────────────────────────
    run_segmentation: bool = False
    segmentation_strategy: SegmentationStrategy = SegmentationStrategy.SKIP

    # ── Background ────────────────────────────────────────────────────────────
    background_strategy: BackgroundStrategy = BackgroundStrategy.NONE
    background_color_hex: Optional[str] = None        # For COLOR_FILL
    background_scene_prompt: Optional[str] = None     # For SD_GENERATION / PROCEDURAL

    # ── Pipeline feature flags ─────────────────────────────────────────────────
    enable_face_preservation: bool = False
    enable_shadow: bool = True
    enable_harmonization: bool = False                # Off by default; on for SD scene BG
    enable_preprocessing: bool = True
    enable_matting: bool = True

    # ── Style ──────────────────────────────────────────────────────────────────
    style_preset: Optional[str] = None               # Drives SD inpainting style
    style_enrichment: Optional[str] = None           # Appended to background SD prompt

    # ── SD safety guards (explicit approval required) ──────────────────────────
    sd_generation_approved: bool = False             # SD background generation
    sd_inpainting_approved: bool = False             # SD foreground inpainting

    # ── Strict Routing and Control properties (mandated) ────────────────────────
    background_type: BackgroundType = BackgroundType.NONE
    use_local_inpainting: bool = False
    use_background_generation: bool = False
    use_color_compositing: bool = False
    use_diffusion: bool = False
    use_face_preservation: bool = False
    use_alpha_matting: bool = False
    preserve_all_people: bool = True
    remove_people: bool = False

    # ── Diagnostics ────────────────────────────────────────────────────────────
    reasoning: List[str] = field(default_factory=list)


# ── Keyword sets (no hardcoded objects — keyword-driven only) ──────────────────

# Backgrounds that are fundamentally non-generative (solid/flat/studio)
_PROCEDURAL_SCENE_KEYWORDS = frozenset([
    "white", "black", "blue", "red", "green", "grey", "gray",
    "yellow", "orange", "purple", "pink", "beige", "ivory",
    "solid", "plain", "flat", "clean", "neutral",
    "studio", "passport", "professional", "corporate", "business",
    "blurred", "blur", "bokeh", "out of focus",
])

# Backgrounds that require actual image synthesis
_GENERATIVE_SCENE_KEYWORDS = frozenset([
    "office", "workspace", "meeting room",
    "school", "classroom", "university", "college", "campus",
    "educational", "academic", "institution", "academy",
    "lecture hall", "auditorium", "laboratory", "lab",
    "beach", "ocean", "sea", "coast",
    "forest", "jungle", "woods", "trees",
    "city", "urban", "street", "downtown", "skyline",
    "mountain", "hill", "valley",
    "space", "galaxy", "stars", "cosmos",
    "kitchen", "dining room", "living room", "bedroom",
    "library", "bookshelf",
    "nature", "outdoor", "outside",
    "indoor", "inside",
    "room", "hall", "corridor",
    "landscape", "scenery", "scene", "environment",
    "park", "garden", "field", "meadow",
    "sunset", "sunrise", "golden hour", "dusk",
    "night", "evening",
    "rainy", "snowy", "foggy",
])

# Style keywords that imply image synthesis is needed
_GENERATIVE_STYLE_KEYWORDS = frozenset([
    "fantasy", "sci-fi", "science fiction", "cyberpunk",
    "anime", "cartoon", "illustration",
    "neon", "futuristic",
    "steampunk", "dystopian", "apocalyptic",
    "medieval", "ancient",
])

# Style → canonical preset name mapping
_STYLE_PRESET_MAP = {
    "passport":          "Passport",
    "passport photo":    "Passport",
    "passport style":    "Passport",
    "id photo":          "Passport",
    "cinematic":         "Cinematic",
    "cinematic look":    "Cinematic",
    "cinematic style":   "Cinematic",
    "film":              "Cinematic",
    "film noir":         "Cinematic",
    "cyberpunk":         "Cyberpunk",
    "cyberpunk neon":    "Cyberpunk",
    "anime":             "Anime",
    "anime style":       "Anime",
    "sci-fi":            "Sci-Fi",
    "science fiction":   "Sci-Fi",
    "fantasy":           "Fantasy",
    "fantasy world":     "Fantasy",
    "professional":      "Professional",
    "professional headshot": "Professional",
    "corporate":         "Professional",
    "business":          "Professional",
    "studio":            "Studio",
    "studio lighting":   "Studio",
    "studio look":       "Studio",
    "realistic":         "Realistic",
    "photorealistic":    "Realistic",
    "photo realistic":   "Realistic",
    "vintage":           "Vintage",
    "vintage look":      "Vintage",
    "retro":             "Vintage",
    "retro style":       "Vintage",
    "neon":              "Neon",
    "neon lights":       "Neon",
    "futuristic":        "Futuristic",
}

# Style → background prompt enrichment suffix
_STYLE_ENRICHMENT_MAP = {
    "Cinematic":    "cinematic shot, anamorphic lens flare, film grain, depth of field",
    "Cyberpunk":    "neon lights, rain-slicked streets, futuristic city, dark atmosphere",
    "Anime":        "anime art style, cel shading, vivid saturated colors, soft lines",
    "Sci-Fi":       "futuristic environment, advanced technology, sleek surfaces, blue glow",
    "Fantasy":      "ethereal lighting, magical atmosphere, mystical particles, lush environment",
    "Vintage":      "film grain, warm tones, soft vignette, aged look",
    "Neon":         "neon glow, vibrant colors, dark background, reflections",
    "Futuristic":   "sleek futuristic design, metallic surfaces, ambient light",
}

# Style descriptors that imply portrait / face preservation
_PORTRAIT_STYLE_SIGNALS = frozenset([
    "passport", "passport photo", "passport style",
    "id photo", "id card style",
    "professional", "professional headshot",
    "corporate", "business",
    "studio", "studio lighting", "studio look",
    "headshot",
])


# ── Internal planner helpers ───────────────────────────────────────────────────

def _is_generative_scene(
    background_request: Optional[str],
    style_descriptors: List[str],
) -> bool:
    """
    Determine whether the background request actually requires image synthesis.

    Returns True ONLY when the scene description implies real-world or fantastical
    image content (office, beach, cyberpunk city) that cannot be produced by a
    procedural gradient or solid fill.

    Conservative by design — ambiguous cases return False (procedural fallback).
    """
    combined = " ".join(
        ([background_request] if background_request else []) + style_descriptors
    ).lower()

    # If any procedural/solid keyword is present → not generative
    if any(kw in combined for kw in _PROCEDURAL_SCENE_KEYWORDS):
        return False

    # If a generative scene keyword is present → generative
    if any(kw in combined for kw in _GENERATIVE_SCENE_KEYWORDS):
        return True

    # If a style that requires synthesis is present → generative
    if any(kw in combined for kw in _GENERATIVE_STYLE_KEYWORDS):
        return True

    # Default: do NOT fire SD (conservative)
    return False


def _resolve_style_preset(style_descriptors: List[str]) -> Optional[str]:
    """
    Map style descriptor strings to a canonical style preset name.
    Longest-match first. Returns None if no match.
    """
    for desc in style_descriptors:
        d = desc.lower()
        # Check from longest key to shortest
        for key in sorted(_STYLE_PRESET_MAP, key=len, reverse=True):
            if key in d:
                return _STYLE_PRESET_MAP[key]
    return None


def _resolve_style_enrichment(style_preset: Optional[str]) -> Optional[str]:
    """Return a prompt enrichment suffix for SD background generation, if applicable."""
    if not style_preset:
        return None
    return _STYLE_ENRICHMENT_MAP.get(style_preset)


def _implies_portrait(style_descriptors: List[str]) -> bool:
    """True when style descriptors suggest a portrait / professional headshot context."""
    combined = " ".join(style_descriptors).lower()
    return any(sig in combined for sig in _PORTRAIT_STYLE_SIGNALS)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_execution_plan(
    intent: ParsedIntent,
    has_custom_background: bool = False,
    use_sd_background: bool = True,
    current_style_preset: str = "Realistic",
) -> ExecutionPlan:
    """
    Convert a ParsedIntent into a safe, deterministic ExecutionPlan.

    Decision rules are applied in priority order:
        1. Object removal → LaMa inpainting (never SD for removal)
        2. Custom background → direct resize (no generation)
        3. Transparency → alpha extraction (no generation)
        4. Solid color → NumPy fill (no generation, SD explicitly blocked)
        5. Scene description → procedural OR SD (depends on content analysis)
        6. Style → preset mapping (SD inpainting only when segmentation runs)
        7. Portrait / face preservation → enable matting + face pipeline
        8. Harmonization guard → disabled for solid/transparent backgrounds

    Args:
        intent:               ParsedIntent from parse_prompt()
        has_custom_background: True when the API request includes a custom BG image
        use_sd_background:    Master switch — user/request level SD permission
        current_style_preset: Fallback style preset from the original request

    Returns:
        ExecutionPlan ready for the orchestrator to consume.
    """
    plan = ExecutionPlan()

    # ── Rule 1: Object Removal ─────────────────────────────────────────────────
    # Any object removal uses LaMa inpainting (localized, non-destructive).
    # SD inpainting is never used for object removal — only for stylistic edits.
    if intent.remove_targets:
        plan.run_object_removal = True
        plan.removal_targets = list(intent.remove_targets)
        plan.foreground_strategy = ForegroundStrategy.LAMA_INPAINTING
        plan.reasoning.append(
            f"Object removal requested → LaMa inpainting for: {intent.remove_targets}"
        )

    # ── Rule 2: Background Strategy ────────────────────────────────────────────

    if has_custom_background:
        # User supplied a background image — no generation needed at all
        plan.background_strategy = BackgroundStrategy.CUSTOM_IMAGE
        plan.run_segmentation = True
        plan.segmentation_strategy = SegmentationStrategy.AUTO_YOLO
        plan.enable_harmonization = True   # Harmonize custom image tones
        plan.reasoning.append("Custom uploaded background image → resize + composite.")

    elif intent.transparency_requested or intent.background_color == "transparent":
        # Transparent background — segmentation + alpha extraction, zero SD
        plan.background_strategy = BackgroundStrategy.TRANSPARENT_FILL
        plan.run_segmentation = True
        plan.segmentation_strategy = SegmentationStrategy.AUTO_YOLO
        plan.enable_matting = True
        plan.enable_shadow = False         # No shadow on transparent
        plan.enable_harmonization = False  # No harmonization needed
        plan.sd_generation_approved = False
        plan.reasoning.append(
            "Transparent background → segmentation + alpha channel. SD blocked."
        )

    elif intent.background_color:
        # ── SAFETY CRITICAL ────────────────────────────────────────────────────
        # A solid color was specified. This is NEVER a generative task.
        # SD is explicitly blocked regardless of use_sd_background flag.
        plan.background_strategy = BackgroundStrategy.COLOR_FILL
        plan.background_color_hex = intent.background_color
        plan.run_segmentation = True
        plan.segmentation_strategy = SegmentationStrategy.AUTO_YOLO
        plan.enable_harmonization = False  # Solid color → no harmonization
        plan.enable_shadow = True
        plan.sd_generation_approved = False   # EXPLICIT SAFETY GUARD
        plan.reasoning.append(
            f"Solid color background ({intent.background_color}) → "
            f"NumPy fill. SD generation BLOCKED."
        )

    elif intent.background_request:
        # Scene description — determine whether it needs real synthesis
        plan.run_segmentation = True
        plan.segmentation_strategy = SegmentationStrategy.AUTO_YOLO

        needs_generation = _is_generative_scene(
            intent.background_request, intent.style_descriptors
        )

        if needs_generation and use_sd_background:
            # Resolve style enrichment before building scene prompt
            style_preset = _resolve_style_preset(intent.style_descriptors)
            enrichment = _resolve_style_enrichment(style_preset)

            scene_prompt = intent.background_request
            if enrichment:
                scene_prompt = f"{scene_prompt}, {enrichment}"

            plan.background_strategy = BackgroundStrategy.SD_GENERATION
            plan.background_scene_prompt = scene_prompt
            plan.style_enrichment = enrichment
            plan.sd_generation_approved = True    # Explicitly approved
            plan.enable_harmonization = True      # Blend subject into generated scene
            plan.reasoning.append(
                f"Generative scene background → SD approved. "
                f"Prompt: '{scene_prompt[:60]}…'"
            )
        else:
            # Ambiguous or procedural scene — use keyword-gradient fallback
            plan.background_strategy = BackgroundStrategy.PROCEDURAL
            plan.background_scene_prompt = intent.background_request
            plan.sd_generation_approved = False
            plan.enable_harmonization = False
            reason = (
                "non-generative keywords detected"
                if not needs_generation
                else "SD disabled by request flag"
            )
            plan.reasoning.append(
                f"Procedural background ({reason}): '{intent.background_request}'."
            )

    # ── Rule 2.5: Portrait/passport style implies solid white background ─────────
    # When a portrait-type style (passport, headshot, professional) is requested
    # and no explicit background instruction was given, default to a white solid fill.
    # This prevents SD from being triggered for common portrait requests.
    _portrait_style = _resolve_style_preset(intent.style_descriptors)
    _passport_styles = {"Passport", "Professional", "Studio"}
    if (
        _portrait_style in _passport_styles
        and plan.background_strategy == BackgroundStrategy.NONE
        and not intent.background_request
        and not intent.background_color
        and not intent.transparency_requested
        and not has_custom_background
    ):
        plan.background_strategy = BackgroundStrategy.COLOR_FILL
        plan.background_color_hex = "#FFFFFF"
        plan.run_segmentation = True
        plan.segmentation_strategy = SegmentationStrategy.AUTO_YOLO
        plan.enable_harmonization = False
        plan.sd_generation_approved = False   # EXPLICIT SAFETY GUARD
        plan.reasoning.append(
            f"Portrait style '{_portrait_style}' → implicit white background. SD BLOCKED."
        )

    # ── Rule 3: Foreground Style / SD Inpainting ───────────────────────────────
    # SD inpainting on the foreground is approved ONLY when:
    #   - segmentation is already running (subject has been isolated)
    #   - the user has NOT explicitly asked to preserve the foreground
    #   - the style preset is a known stylistic transform (not just "Realistic")
    style_preset = _resolve_style_preset(intent.style_descriptors) or current_style_preset
    plan.style_preset = style_preset

    if (
        style_preset
        and style_preset.lower() not in ("realistic", "professional", "passport", "studio")
        and plan.run_segmentation
        and not intent.preserve_foreground
        and not intent.remove_targets   # Don't mix LaMa removal + SD inpainting
    ):
        plan.foreground_strategy = ForegroundStrategy.SD_INPAINTING
        plan.sd_inpainting_approved = True
        plan.reasoning.append(
            f"SD inpainting approved for style '{style_preset}' on isolated foreground."
        )
    elif style_preset:
        plan.reasoning.append(
            f"Style preset '{style_preset}' noted; foreground preserved (no SD inpainting)."
        )

    # ── Rule 4: Face / Identity Preservation ──────────────────────────────────
    if intent.preserve_identity or _implies_portrait(intent.style_descriptors):
        plan.enable_face_preservation = True
        plan.enable_matting = True
        plan.reasoning.append(
            "Face/identity preservation enabled (portrait or explicit preserve signal)."
        )

    # ── Rule 5: Foreground preservation override ───────────────────────────────
    if intent.preserve_foreground:
        plan.foreground_strategy = ForegroundStrategy.PRESERVE_PIXELS
        plan.sd_inpainting_approved = False
        plan.reasoning.append(
            "Foreground preservation explicitly requested → SD inpainting BLOCKED."
        )

    # ── Rule 6: Harmonization safety guard ────────────────────────────────────
    # Color harmonization makes no sense over solid, transparent, or absent backgrounds.
    # Override regardless of what was set above.
    if plan.background_strategy in (
        BackgroundStrategy.COLOR_FILL,
        BackgroundStrategy.TRANSPARENT_FILL,
        BackgroundStrategy.NONE,
    ):
        plan.enable_harmonization = False

    # ── Rule 7: Segmentation implies matting ──────────────────────────────────
    if plan.run_segmentation:
        plan.enable_matting = True

    # ── Strict SD Safety Override ──────────────────────────────────────────
    # Stable Diffusion MUST ONLY execute IF background_type == GENERATED_SCENE
    plan.background_type = intent.background_type or BackgroundType.NONE
    if plan.background_type != BackgroundType.GENERATED_SCENE:
        plan.sd_generation_approved = False
        plan.sd_inpainting_approved = False
        plan.foreground_strategy = ForegroundStrategy.PRESERVE_PIXELS
        plan.reasoning.append("Stable Diffusion blocked: background is not a generated scene.")

    # Populate final execution properties
    plan.use_local_inpainting = bool(intent.remove_targets)
    plan.use_background_generation = (plan.background_type == BackgroundType.GENERATED_SCENE)
    plan.use_color_compositing = (plan.background_type in (BackgroundType.SOLID_COLOR, BackgroundType.TRANSPARENT, BackgroundType.STUDIO))
    plan.use_diffusion = plan.sd_generation_approved or plan.sd_inpainting_approved
    plan.use_face_preservation = plan.enable_face_preservation
    plan.use_alpha_matting = plan.enable_matting

    # Detect if any removal target is a person/human
    person_kws = {"person", "man", "woman", "girl", "boy", "guy", "child", "baby", "people", "someone"}
    has_person_removal = False
    for target in intent.remove_targets:
        target_lower = target.lower()
        if any(kw in target_lower for kw in person_kws):
            has_person_removal = True
            break

    plan.remove_people = has_person_removal
    plan.preserve_all_people = not has_person_removal

    return plan

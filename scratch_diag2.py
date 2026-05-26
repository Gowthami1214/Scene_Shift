"""
Extended diagnostic — finds ALL routing failures across many prompt patterns.
Run from project root: python scratch_diag2.py
"""
import sys, os, types, importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

BASE = r"c:\Users\Gowthami\Computer Vision project"
src_pkg = types.ModuleType("src")
src_pkg.pipeline = types.ModuleType("src.pipeline")
sys.modules["src"] = src_pkg
sys.modules["src.pipeline"] = src_pkg.pipeline

pp = load_module("src.pipeline.prompt_parser",
                 os.path.join(BASE, "src", "pipeline", "prompt_parser.py"))
sys.modules["src.pipeline.prompt_parser"] = pp
ep = load_module("src.pipeline.execution_planner",
                 os.path.join(BASE, "src", "pipeline", "execution_planner.py"))

parse_prompt = pp.parse_prompt
build_execution_plan = ep.build_execution_plan
_extract_background = pp._extract_background

# (prompt, expected_strategy)
CASES = [
    # Simple color patterns — all must be color_fill
    ("change background to blue",       "color_fill"),
    ("make background red",             "color_fill"),
    ("white background",                "color_fill"),
    ("solid green background",          "color_fill"),
    ("background to orange",            "color_fill"),
    ("set the background to dark navy", "color_fill"),
    ("make it a red background",        "color_fill"),
    ("blue background please",          "color_fill"),
    ("i want white background",         "color_fill"),
    ("use a solid blue background",     "color_fill"),

    # Subject + color phrasing variants
    ("background should be green",      "color_fill"),   # BUG: 'should be'
    ("background needs to be white",    "color_fill"),   # BUG: 'needs to be'
    ("background must be black",        "color_fill"),   # BUG: 'must be'
    ("give me a blue background",       "color_fill"),
    ("i need a white background",       "color_fill"),
    ("can you change the bg to red",    "color_fill"),
    ("background colour blue",          "color_fill"),   # BUG: 'colour' spelling
    ("bg blue",                         "color_fill"),
    ("bg should be white",              "color_fill"),   # BUG
    ("background = white",              "color_fill"),   # BUG
    ("background: blue",                "color_fill"),
    ("background — white",              "color_fill"),   # BUG: em-dash
    ("background is red",               "color_fill"),   # BUG: 'is'

    # These should be generative scenes — must NOT be color_fill
    ("replace background with busy city office",  "sd_generation"),
    ("cyberpunk city background",                 "sd_generation"),
    ("beach sunset background",                   "sd_generation"),
    ("forest background",                         "sd_generation"),

    # Transparent
    ("transparent background",          "transparent_fill"),
    ("remove background",               "transparent_fill"),
]

bugs = []
for prompt, expected in CASES:
    intent = parse_prompt(prompt)
    plan   = build_execution_plan(intent, use_sd_background=True)
    actual = plan.background_strategy.value

    ok = (actual == expected)
    status = "OK  " if ok else "BUG "
    if not ok:
        bugs.append((prompt, expected, actual,
                     intent.background_request, intent.background_color))
    print(f"[{status}] {expected:20s} | got={actual:20s} | \"{prompt}\"")

print()
print(f"{'='*65}")
print(f"Bugs found: {len(bugs)}/{len(CASES)}")
for prompt, exp, got, req, col in bugs:
    print(f"  FAIL: \"{prompt}\"")
    print(f"        expected={exp!r}, got={got!r}")
    print(f"        extracted request={req!r}, color={col!r}")

"""
Diagnostic: trace exact routing for reported failing prompts.
Run from project root: python scratch_diag.py
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
find_solid_color = pp.find_solid_color
build_execution_plan = ep.build_execution_plan
_extract_background = pp._extract_background

FAILING_PROMPTS = [
    "change background to blue",
    "make background red",
    "white background",
    "solid green background",
    "background to orange",
    "set the background to dark navy",
    "make it a red background",
    "blue background please",
    "i want white background",
    "background should be green",
    "use a solid blue background",
]

print("=" * 65)
print("DIAGNOSTIC: Background Routing Trace")
print("=" * 65)
for prompt in FAILING_PROMPTS:
    intent = parse_prompt(prompt)
    plan   = build_execution_plan(intent, use_sd_background=True)

    bg_req, bg_col = _extract_background(prompt)

    status = "OK" if plan.background_strategy.value == "color_fill" else "BUG"
    print(f"\n[{status}] \"{prompt}\"")
    print(f"       _extract_background -> request={bg_req!r}, color={bg_col!r}")
    print(f"       intent.background_request = {intent.background_request!r}")
    print(f"       intent.background_color   = {intent.background_color!r}")
    print(f"       plan.background_strategy  = {plan.background_strategy.value!r}")
    print(f"       plan.sd_generation_approved = {plan.sd_generation_approved}")
    if status == "BUG":
        print(f"       plan.reasoning = {plan.reasoning}")

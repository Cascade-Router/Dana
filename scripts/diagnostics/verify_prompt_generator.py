"""Standalone smoke test for the prompt-generation/patching module.

Exercises dana.prompts.spatial_synthesis.build_agent_system_prompt (the pure
string-assembly layer) and, if importable without side effects, the runtime
wrapper dana.core_agent.build_dana_system_prompt.

Run: python scripts/diagnostics/verify_prompt_generator.py
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def check_spatial_synthesis():
    from dana.prompts.spatial_synthesis import build_agent_system_prompt

    cases = [
        dict(spatial_block="", labels_csv="", profile_summary="No long-term user profile stored yet.",
             reply_lang="en", timezone=None, home_city=None, home_region=None, vault_hot_cache=None),
        dict(spatial_block="SPATIAL_IR: {}", labels_csv="button,textbox", profile_summary="User is Alex.",
             reply_lang="fa", timezone="America/Los_Angeles", home_city="Sunnyvale", home_region="CA",
             vault_hot_cache={"user_name": "Alex", "family_partner": "Sam"}),
        dict(spatial_block="", labels_csv="", profile_summary="", reply_lang="mixed"),
    ]

    for i, kwargs in enumerate(cases, 1):
        prompt = build_agent_system_prompt(**kwargs)
        assert isinstance(prompt, str) and prompt.strip(), f"case {i}: empty/non-str prompt"
        print(f"  case {i} ({kwargs['reply_lang']}): OK, {len(prompt)} chars")
    return True


def check_core_agent_wrapper():
    from dana.core.agent_loop import build_dana_system_prompt

    prompt = build_dana_system_prompt(yolo_labels=[], profile=None, user_text="hello")
    assert isinstance(prompt, str) and prompt.strip(), "empty/non-str prompt from wrapper"
    print(f"  wrapper OK, {len(prompt)} chars")
    return True


def main():
    results = {}

    print("[1/2] dana.prompts.spatial_synthesis.build_agent_system_prompt")
    try:
        results["spatial_synthesis"] = check_spatial_synthesis()
    except Exception:
        results["spatial_synthesis"] = False
        traceback.print_exc()

    print("[2/2] dana.core_agent.build_dana_system_prompt (full runtime wrapper)")
    try:
        results["core_agent_wrapper"] = check_core_agent_wrapper()
    except Exception:
        results["core_agent_wrapper"] = False
        traceback.print_exc()

    print("\n--- summary ---")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

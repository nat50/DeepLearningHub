"""
Generate website data artifacts for the DeepLearningHub site.

This script is intentionally kept small. Assignment-specific logic lives in
dedicated generator modules so future assignments can add new pipelines without
growing this file into a monolith.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from pathlib import Path

GeneratorFn = Callable[[], list[Path]]


def generate_assignment1_image_classification() -> list[Path]:
    from website_data.generators.assignment1.image_classification import (
        generate_assignment1_image_classification_website_data,
    )

    return generate_assignment1_image_classification_website_data()


def generate_assignment1_text_classification() -> list[Path]:
    from website_data.generators.assignment1.text_classification import (
        generate_assignment1_text_classification_website_data,
    )

    return generate_assignment1_text_classification_website_data()


def generate_assignment1_multimodal_classification() -> list[Path]:
    from website_data.generators.assignment1.multimodal_classification import (
        generate_assignment1_multimodal_classification_website_data,
    )

    return generate_assignment1_multimodal_classification_website_data()

GENERATOR_REGISTRY: dict[str, dict[str, GeneratorFn]] = {
    "assignment1": {
        "image_classification": generate_assignment1_image_classification,
        "text_classification": generate_assignment1_text_classification,
        "multimodal_classification": generate_assignment1_multimodal_classification,
    },
    "assignment2": {},
    "assignment3": {},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate website data artifacts for one or more assignments.")
    parser.add_argument(
        "--assignment",
        action="append",
        choices=sorted(GENERATOR_REGISTRY),
        help="Run generators only for the selected assignment. Repeat to select multiple assignments.",
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        help="Run only the named pipeline(s) within the selected assignment(s), for example: image_classification",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all registered generators and exit.",
    )
    return parser.parse_args()


def iter_selected_generators(
    assignments: list[str] | None,
    pipelines: list[str] | None,
) -> Iterable[tuple[str, str, GeneratorFn]]:
    selected_assignments = assignments or list(GENERATOR_REGISTRY)
    selected_pipelines = set(pipelines or [])

    for assignment in selected_assignments:
        assignment_generators = GENERATOR_REGISTRY.get(assignment, {})
        for pipeline_name, generator_fn in assignment_generators.items():
            if selected_pipelines and pipeline_name not in selected_pipelines:
                continue
            yield assignment, pipeline_name, generator_fn


def print_registry() -> None:
    print("Registered website-data generators:")
    for assignment, pipelines in GENERATOR_REGISTRY.items():
        if not pipelines:
            print(f"- {assignment}: (no generators registered yet)")
            continue
        for pipeline_name in sorted(pipelines):
            print(f"- {assignment}.{pipeline_name}")


def main() -> None:
    args = parse_args()
    if args.list:
        print_registry()
        return

    selected_generators = list(iter_selected_generators(args.assignment, args.pipeline))
    if not selected_generators:
        raise SystemExit("No generators matched the selected assignment/pipeline filters.")

    print("=" * 60)
    print("Generating website data for DeepLearningHub")
    print("=" * 60)

    generated_files: list[Path] = []
    for assignment, pipeline_name, generator_fn in selected_generators:
        print(f"Running {assignment}.{pipeline_name} ...")
        generated_files.extend(generator_fn())

    print("=" * 60)
    print(f"Done! Generated {len(generated_files)} file(s) across {len(selected_generators)} pipeline(s).")


if __name__ == "__main__":
    main()

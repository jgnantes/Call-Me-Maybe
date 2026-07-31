import argparse
import sys
import llm_sdk as llm
from . import interpretation_utils as iu
from . import json_utils as ju


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert prompts into structured function calls."
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the prompt input JSON file.",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the function calling pipeline with parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code.
    """
    function_file = ju.load_function_definition_file(
        args.functions_definition
    )
    prompt_file = ju.load_prompt_input_file(args.input)

    model = getattr(llm, "Small_LLM_Model")()

    results = [
        iu.interpret_prompt(
            prompt_input.prompt,
            function_file.functions,
            model,
        )
        for prompt_input in prompt_file.prompts
    ]

    ju.dump_function_call_results(args.output, results)

    print(f"Wrote {len(results)} function calls to {args.output}")
    return 0


def main() -> int:
    """Run the CLI and report failures with clean messages.

    Returns:
        Process exit code.
    """
    try:
        return run_pipeline(parse_args())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import json
from pathlib import Path
from typing import Any
from pydantic import ValidationError
from models import (
    FunctionCallResult,
    FunctionDefinitionFile,
    PromptInputFile,
)


def load_json_file(path: str) -> Any:
    """Load raw JSON data from a file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain valid JSON.
        OSError: If the file cannot be read.
    """
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON file: {path}") from error


def load_prompt_input_file(path: str) -> PromptInputFile:
    """Load and validate prompt inputs from a JSON file.

    Args:
        path: Path to the prompt input JSON file.

    Returns:
        Validated prompt input file model.

    Raises:
        ValueError: If the JSON structure does not match the expected schema.
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    data = load_json_file(path)
    try:
        return PromptInputFile(prompts=data)
    except ValidationError as error:
        raise ValueError(f"invalid prompt input file: {path}") from error


def load_function_definition_file(path: str) -> FunctionDefinitionFile:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the function definitions JSON file.

    Returns:
        Validated function definition file model.

    Raises:
        ValueError: If the JSON structure does not match the expected schema.
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    data = load_json_file(path)
    try:
        return FunctionDefinitionFile(functions=data)
    except ValidationError as error:
        raise ValueError(f"invalid function definition file: {path}") from error


def dump_function_call_results(
        path: str,
        results: list[FunctionCallResult],
) -> None:
    """Write function call results to a JSON file.

    Args:
        path: Path where the output JSON file should be written.
        results: Function call results to serialize.

    Raises:
        OSError: If the output file cannot be written.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        result.model_dump()
        for result in results
    ]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


if __name__=="__main__":
    input = load_prompt_input_file("data/input/function_calling_tests.json")
    functions = load_function_definition_file(
        "data/input/functions_definition.json")
    result = FunctionCallResult(
        prompt="what is 2 plus 2 equal to?",
        name="fn_add_numbers",
        parameters={"a": 2, "b": 2},
    )
    dump_function_call_results(
        "data/output/test.json",
        [result],
    )
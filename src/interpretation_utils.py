import re
import llm_sdk as llm
from typing import Any
from decoding_utils import (
    choose_from_allowed_texts,
    choose_function,
)
from models import (
    FunctionCallResult,
    FunctionDefinition,
    validate_call_against_definition,
)


def extract_words_from_prompt(prompt: str) -> list[str]:
    """Extract word-like candidates from a natural-language prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Word candidates found in the prompt, preserving their order.
    """
    return re.findall(r"[A-Za-z_]+", prompt)


def extract_numbers_from_prompt(prompt: str) -> list[int | float]:
    """Extract numeric values from a natural-language prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Numeric values found in the prompt, preserving their order.
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?", prompt)
    numbers: list[int | float] = []

    for match in matches:
        if "." in match:
            numbers.append(float(match))
        else:
            numbers.append(int(match))

    return numbers


def extract_quoted_strings_from_prompt(prompt: str) -> list[str]:
    """Extract quoted strings from a natural-language prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Strings found between single or double quotes.
    """
    return re.findall(r"""["']([^"']*)["']""", prompt)


def build_string_argument_prompt(
        prompt: str,
        function: FunctionDefinition,
        parameter_name: str,
) -> str:
    """Build a prompt for selecting a string argument value.

    Args:
        prompt: Natural-language request containing the argument.
        function: Function definition that expects the argument.
        parameter_name: Name of the parameter to extract.

    Returns:
        Prompt asking the model to select the parameter value.
    """
    return (
        "Select the value for the function parameter.\n"
        f"User request: {prompt}\n"
        f"Function: {function.name}\n"
        f"Function description: {function.description}\n"
        f"Parameter: {parameter_name}\n"
        "Value:"
    )


def choose_string_argument(
        prompt: str,
        function: FunctionDefinition,
        parameter_name: str,
        candidates: list[str],
        model: llm.Small_LLM_Model,
) -> str:
    """Choose a string argument from candidate words.

    Args:
        prompt: Natural-language request containing the argument.
        function: Function definition that expects the argument.
        parameter_name: Name of the string parameter to extract.
        candidates: Candidate string values found in the prompt.
        model: LLM wrapper used for constrained decoding.

    Returns:
        Selected string argument.

    Raises:
        ValueError: If no candidate values are available.
    """
    if not candidates:
        raise ValueError(f"missing string candidates for {parameter_name}")

    argument_prompt = build_string_argument_prompt(
        prompt,
        function,
        parameter_name,
    )
    allowed_texts = [
        f" {candidate}"
        for candidate in candidates
    ]

    return choose_from_allowed_texts(
        argument_prompt,
        allowed_texts,
        model,
    ).strip()


def extract_parameters(
        prompt: str,
        function: FunctionDefinition,
        model: llm.Small_LLM_Model,
) -> dict[str, Any]:
    """Extract function parameters from a natural-language prompt.

    Args:
        prompt: Natural-language request to convert into parameters.
        function: Function definition selected for the prompt.
        model: LLM wrapper available for future constrained extraction.

    Returns:
        Extracted parameter values keyed by parameter name.

    Raises:
        ValueError: If parameters cannot be extracted from the prompt.
    """
    parameters: dict[str, Any] = {}
    numbers = extract_numbers_from_prompt(prompt)
    quoted_strings = extract_quoted_strings_from_prompt(prompt)

    number_index = 0
    string_index = 0

    for parameter_name, parameter_definition in function.parameters.items():
        if parameter_definition.type == "number":
            if number_index >= len(numbers):
                raise ValueError(f"missing number for {parameter_name}")
            parameters[parameter_name] = numbers[number_index]
            number_index += 1

        if parameter_definition.type == "string":
            if string_index < len(quoted_strings):
                parameters[parameter_name] = quoted_strings[string_index]
                string_index += 1
            else:
                words = extract_words_from_prompt(prompt)
                parameters[parameter_name] = choose_string_argument(
                    prompt,
                    function,
                    parameter_name,
                    words,
                    model,
                )

    return parameters


def build_function_call_result(
        prompt: str,
        function: FunctionDefinition,
        model: llm.Small_LLM_Model,
) -> FunctionCallResult:
    """Build a validated function call result for a prompt.

    Args:
        prompt: Original natural-language request.
        function: Function definition selected for the prompt.
        model: LLM wrapper used for argument extraction.

    Returns:
        Validated function call result.

    Raises:
        ValueError: If extracted parameters do not match the function schema.
    """
    parameters = extract_parameters(prompt, function, model)
    result = FunctionCallResult(
        prompt=prompt,
        name=function.name,
        parameters=parameters,
    )
    validate_call_against_definition(result, function)
    return result


def interpret_prompt(
        prompt: str,
        functions: list[FunctionDefinition],
        model: llm.Small_LLM_Model,
) -> FunctionCallResult:
    """Interpret a prompt as a validated function call.

    Args:
        prompt: Natural-language request to convert into a function call.
        functions: Available function definitions.
        model: LLM wrapper used for function selection and argument extraction.

    Returns:
        Validated function call result.
    """
    function = choose_function(prompt, functions, model)
    return build_function_call_result(prompt, function, model)

import re
from typing import Any
import llm_sdk as llm
from .decoding_utils import (
    choose_from_allowed_texts,
    choose_function,
)
from .models import (
    FunctionCallResult,
    FunctionDefinition,
    TypeDefinition,
    validate_call_against_definition,
)


REGEX_PATTERN_CANDIDATES = [
    r"\d+",
    r"\D+",
    r"[A-Z]",
    r"[a-z]",
    r"[aeiouAEIOU]",
    r"\w+",
    r"\s+",
]


def deduplicate_preserving_order(values: list[str]) -> list[str]:
    """Remove duplicate strings while preserving their first occurrence.

    Args:
        values: String values to deduplicate.

    Returns:
        Deduplicated values in their original order.
    """
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        clean_value = value.strip()
        if clean_value and clean_value not in seen:
            seen.add(clean_value)
            result.append(clean_value)

    return result


def extract_words_from_prompt(prompt: str) -> list[str]:
    """Extract word-like candidates from a natural-language prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Word candidates found in the prompt, preserving their order.
    """
    return re.findall(r"[A-Za-z_]+", prompt)


def extract_number_texts_from_prompt(prompt: str) -> list[str]:
    """Extract numeric strings from a natural-language prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Numeric strings found in the prompt, preserving their order.
    """
    return re.findall(r"-?\d+(?:\.\d+)?", prompt)


def extract_quoted_strings_from_prompt(prompt: str) -> list[str]:
    """Extract quoted strings from a natural-language prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Strings found between matching single or double quotes.
    """
    matches = re.findall(r"""(["'])(.*?)\1""", prompt)
    return [
        match[1]
        for match in matches
    ]


def collect_text_candidates(prompt: str) -> list[str]:
    """Collect text candidates from a prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Candidate text values found in the prompt.
    """
    words = prompt.split()
    candidates: list[str] = []

    candidates.extend(extract_quoted_strings_from_prompt(prompt))
    candidates.extend(extract_number_texts_from_prompt(prompt))
    candidates.extend(extract_words_from_prompt(prompt))

    for start_index in range(1, min(len(words), 4)):
        candidates.append(" ".join(words[start_index:]))

    if ":" in prompt:
        candidates.append(prompt.split(":", 1)[1])

    candidates.append(prompt)

    return deduplicate_preserving_order(candidates)


def collect_candidates_for_parameter(
        prompt: str,
        parameter_name: str,
        parameter_definition: TypeDefinition,
        selected_parameters: dict[str, Any],
) -> list[str]:
    """Collect candidates suitable for a specific parameter.

    Args:
        prompt: Natural-language request to inspect.
        parameter_name: Name of the parameter being extracted.
        parameter_definition: Type definition for the parameter.
        selected_parameters: Parameters already selected for this function
            call.

    Returns:
        Candidate values for the parameter.
    """
    if parameter_definition.type == "number":
        numbers = extract_number_texts_from_prompt(prompt)
        used_numbers = {
            str(value)
            for value in selected_parameters.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        unused_numbers = [
            number
            for number in numbers
            if number not in used_numbers
        ]
        return unused_numbers or numbers or ["0"]

    if parameter_definition.type == "boolean":
        return ["true", "false"]

    candidates = collect_text_candidates(prompt)

    if parameter_name == "regex":
        candidates = REGEX_PATTERN_CANDIDATES + candidates

    if "source" in parameter_name:
        candidates = (
            extract_quoted_strings_from_prompt(prompt)
            + candidates
        )

    if "replacement" in parameter_name:
        candidates = extract_words_from_prompt(prompt) + candidates

    if parameter_definition.type == "object":
        candidates = ["{}"] + candidates

    if parameter_definition.type == "array":
        candidates = ["[]"] + candidates

    return deduplicate_preserving_order(candidates)


def build_argument_selection_prompt(
        prompt: str,
        function: FunctionDefinition,
        parameter_name: str,
        parameter_definition: TypeDefinition,
        candidates: list[str],
        selected_parameters: dict[str, Any],
) -> str:
    """Build a prompt for selecting a function argument.

    Args:
        prompt: Natural-language request containing the argument.
        function: Function definition that expects the argument.
        parameter_name: Name of the parameter to extract.
        parameter_definition: Type definition for the parameter.
        candidates: Candidate values available for selection.
        selected_parameters: Parameters already selected for this function
            call.

    Returns:
        Prompt asking the model to select the parameter value.
    """
    candidate_lines = [
        f"- {candidate}"
        for candidate in candidates
    ]
    selected_lines = [
        f"{name} = {value}"
        for name, value in selected_parameters.items()
    ]
    selected_block = "\n".join(selected_lines) or "None"

    return (
        "Select exactly one value for the current function parameter.\n"
        "Use the user request, the selected function, and the already "
        "selected parameters as context.\n\n"
        f"User request:\n{prompt}\n\n"
        f"Selected function:\n{function.name}\n\n"
        f"Function description:\n{function.description}\n\n"
        f"Current parameter:\n{parameter_name}\n\n"
        f"Current parameter type:\n{parameter_definition.type}\n\n"
        f"Already selected parameters:\n{selected_block}\n\n"
        "Allowed values:\n"
        + "\n".join(candidate_lines)
        + "\n\nValue:"
    )


def choose_argument_text(
        prompt: str,
        function: FunctionDefinition,
        parameter_name: str,
        parameter_definition: TypeDefinition,
        candidates: list[str],
        selected_parameters: dict[str, Any],
        model: llm.Small_LLM_Model,
) -> str:
    """Choose an argument value as text using constrained decoding.

    Args:
        prompt: Natural-language request containing the argument.
        function: Function definition that expects the argument.
        parameter_name: Name of the parameter to extract.
        parameter_definition: Type definition for the parameter.
        candidates: Candidate values available for selection.
        selected_parameters: Parameters already selected for this function
            call.
        model: LLM wrapper used for constrained decoding.

    Returns:
        Selected argument value as text.

    Raises:
        ValueError: If no candidates are available.
    """
    if not candidates:
        raise ValueError(f"missing candidates for {parameter_name}")

    argument_prompt = build_argument_selection_prompt(
        prompt,
        function,
        parameter_name,
        parameter_definition,
        candidates,
        selected_parameters,
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


def coerce_argument_value(
        value: str,
        parameter_definition: TypeDefinition,
) -> Any:
    """Convert a selected argument string into its schema type.

    Args:
        value: Selected argument value as text.
        parameter_definition: Type definition expected by the schema.

    Returns:
        Value converted to the expected type.
    """
    if parameter_definition.type == "number":
        return float(value) if "." in value else int(value)

    if parameter_definition.type == "boolean":
        return value.strip().lower() == "true"

    if parameter_definition.type == "object":
        return {}

    if parameter_definition.type == "array":
        return []

    return value


def extract_parameters(
        prompt: str,
        function: FunctionDefinition,
        model: llm.Small_LLM_Model,
) -> dict[str, Any]:
    """Extract function parameters from a natural-language prompt.

    Args:
        prompt: Natural-language request to convert into parameters.
        function: Function definition selected for the prompt.
        model: LLM wrapper used for constrained argument selection.

    Returns:
        Extracted parameter values keyed by parameter name.

    Raises:
        ValueError: If a selected parameter value cannot be coerced.
    """
    parameters: dict[str, Any] = {}

    for parameter_name, parameter_definition in function.parameters.items():
        candidates = collect_candidates_for_parameter(
            prompt,
            parameter_name,
            parameter_definition,
            parameters,
        )
        selected_value = choose_argument_text(
            prompt,
            function,
            parameter_name,
            parameter_definition,
            candidates,
            parameters,
            model,
        )
        parameters[parameter_name] = coerce_argument_value(
            selected_value,
            parameter_definition,
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

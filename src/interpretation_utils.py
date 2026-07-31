import argparse
import json
import re
from typing import Any
import llm_sdk as llm
from .decoding_utils import (
    choose_function,
    generate_boolean,
    generate_number,
    generate_string,
    get_generation_tokens,
    get_input_ids,
)
from .models import (
    FunctionCallResult,
    FunctionDefinition,
    TypeDefinition,
    validate_call_against_definition,
)


def extract_prompt_numbers(prompt: str) -> list[int | float]:
    """Extract numeric literals from a prompt, preserving sign and order.

    Args:
        prompt: Natural-language request that may contain numbers.

    Returns:
        Numbers found in the prompt, ordered by appearance.
    """
    number_texts = re.findall(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])", prompt)
    numbers: list[int | float] = []

    for number_text in number_texts:
        if "." in number_text:
            numbers.append(float(number_text))
        else:
            numbers.append(int(number_text))

    return numbers


def get_ordered_number_parameters(
        function: FunctionDefinition,
) -> list[str]:
    """Get numeric parameter names in definition order.

    Args:
        function: Function definition whose parameters should be inspected.

    Returns:
        Names of parameters typed as numbers.
    """
    return [
        parameter_name
        for parameter_name, parameter_definition
        in function.parameters.items()
        if parameter_definition.type == "number"
    ]


def build_argument_context(prompt: str, function: FunctionDefinition) -> str:
    """Build the shared context used before the partial JSON object.

    Args:
        prompt: Natural-language request to convert into parameters.
        function: Function definition selected for the prompt.

    Returns:
        Context that asks the model to extract arguments as JSON.
    """
    return (
        f'User request: "{prompt}"\n'
        f"Available function: {function.name} - {function.description}\n"
        "Extract the arguments as a JSON object:\n"
    )


def generate_argument_value(
        model: Any,
        context: str,
        parameter_definition: TypeDefinition,
        digit_tokens: dict[str, int],
        quote_tokens: set[int],
) -> tuple[Any, str]:
    """Generate one argument value and its JSON text representation.

    Args:
        model: LLM wrapper used for inference.
        context: Text that appears immediately before the argument value.
        parameter_definition: Type definition expected by the schema.
        digit_tokens: Mapping between numeric characters and token IDs.
        quote_tokens: Token IDs that contain quote characters.

    Returns:
        Generated Python value and JSON text for the partial object.
    """
    if parameter_definition.type == "number":
        number_value = generate_number(
            model,
            get_input_ids(context, model),
            digit_tokens,
        )
        return number_value, str(number_value)

    if parameter_definition.type == "string":
        string_context = context + '"'
        string_value = generate_string(
            model,
            get_input_ids(string_context, model),
            quote_tokens,
        )
        return string_value, json.dumps(string_value, ensure_ascii=False)

    if parameter_definition.type == "boolean":
        boolean_value = generate_boolean(model, context)
        return boolean_value, json.dumps(boolean_value)

    raise ValueError(
        f"unsupported parameter type: {parameter_definition.type}"
    )


def extract_parameters(
        prompt: str,
        function: FunctionDefinition,
        model: Any,
) -> dict[str, Any]:
    """Extract function parameters from a natural-language prompt.

    Args:
        prompt: Natural-language request to convert into parameters.
        function: Function definition selected for the prompt.
        model: LLM wrapper used for constrained argument generation.

    Returns:
        Extracted parameter values keyed by parameter name.
    """
    parameters: dict[str, Any] = {}
    base_context = build_argument_context(prompt, function)
    json_so_far = "{"
    digit_tokens, quote_tokens = get_generation_tokens(model)
    parameter_items = list(function.parameters.items())
    prompt_numbers = extract_prompt_numbers(prompt)
    number_parameters = get_ordered_number_parameters(function)

    for index, (parameter_name, parameter_definition) in enumerate(
            parameter_items,
    ):
        json_so_far += f'"{parameter_name}": '
        if parameter_name in number_parameters:
            number_index = number_parameters.index(parameter_name)
            if number_index < len(prompt_numbers):
                value = prompt_numbers[number_index]
                json_value = str(value)
            else:
                value, json_value = generate_argument_value(
                    model,
                    base_context + json_so_far,
                    parameter_definition,
                    digit_tokens,
                    quote_tokens,
                )
        else:
            value, json_value = generate_argument_value(
                model,
                base_context + json_so_far,
                parameter_definition,
                digit_tokens,
                quote_tokens,
            )
        parameters[parameter_name] = value
        json_so_far += json_value

        if index < len(parameter_items) - 1:
            json_so_far += ", "

    return parameters


def build_function_call_result(
        prompt: str,
        function: FunctionDefinition,
        model: Any,
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
        model: Any,
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


if __name__ == "__main__":
    from .json_utils import load_function_definition_file

    parser = argparse.ArgumentParser(
        description="Test prompts against available function definitions.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to test. Can be used more than once.",
    )
    parser.add_argument(
        "--function",
        action="append",
        default=[],
        help="Optional function name filter. Can be used more than once.",
    )
    parser.add_argument(
        "--functions-definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    args = parser.parse_args()

    function_file = load_function_definition_file(args.functions_definition)
    test_functions = [
        function
        for function in function_file.functions
        if not args.function or function.name in args.function
    ]

    test_prompts = args.prompt or [
        "Disable notifications for Pedro",
        "Enable notifications for Maria",
        "How much is one plus fourty two thousand million"
    ]

    test_model = getattr(llm, "Small_LLM_Model")()

    for test_prompt in test_prompts:
        print("________________________")
        test_result = interpret_prompt(
            test_prompt,
            test_functions,
            test_model,
        )

        print("Prompt:")
        print(test_prompt)
        print()
        print("FunctionCallResult:")
        print(test_result.model_dump())
        print()

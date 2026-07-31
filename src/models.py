from typing import Any, Literal
from pydantic import BaseModel, model_validator, ConfigDict


def validate_non_empty_list(values: list, label: str) -> None:
    """Validate that a list is not empty.

    Args:
        values: List to validate.
        label: Human-readable name used in the error message.

    Raises:
        ValueError: If the list is empty.
    """
    if not values:
        raise ValueError(f"{label} must not be empty")


class PromptInput(BaseModel):
    """Represent a natural-language prompt loaded from the input file.

    Attributes:
        prompt: Natural-language request to convert into a function call.
    """

    prompt: str

    @model_validator(mode='after')
    def validate_empty_prompt(self) -> "PromptInput":
        """Validate that the prompt is not empty.

        Args:
            prompt: Natural-language request to validate.

        Returns:
            The validated prompt.

        Raises:
            ValueError: If the prompt is empty or only whitespace.
        """
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        return self


class PromptInputFile(BaseModel):
    """Represent the full prompt input file.

    Attributes:
        prompts: Prompt entries loaded from the input file.
    """

    prompts: list[PromptInput]

    @model_validator(mode="after")
    def validate_prompt_input_file(self) -> "PromptInputFile":
        """Validate that the prompt input file is not empty.

        Returns:
            The validated prompt input file.
        """
        validate_non_empty_list(self.prompts, "input file")
        return self


class TypeDefinition(BaseModel):
    """Represent a JSON-compatible type definition.

    Attributes:
        type: JSON-compatible type name.
    """

    type: Literal["string", "number"]


class FunctionDefinition(BaseModel):
    """Represent a callable function definition.

    Attributes:
        name: Name of the function.
        description: Natural-language description of the function behavior.
        parameters: Mapping between parameter names and their type definitions.
        returns: Type definition for the function return value.
    """

    name: str
    description: str
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition

    @model_validator(mode="after")
    def validate_function_definition(self) -> "FunctionDefinition":
        """Validate that the function definition is complete.

        Returns:
            The validated function definition.

        Raises:
            ValueError: If required textual fields or parameter names are
                empty.
        """
        if not self.name.strip():
            raise ValueError("function name must not be empty")
        if not self.description.strip():
            raise ValueError("function description must not be empty")
        for parameter_name in self.parameters:
            if not parameter_name.strip():
                raise ValueError("parameter name must not be empty")
        return self


class FunctionDefinitionFile(BaseModel):
    """Represent the full function definitions file.

    Attributes:
        functions: Function definitions loaded from the definitions file.
    """

    functions: list[FunctionDefinition]

    @model_validator(mode="after")
    def validate_function_definition_file(self) -> "FunctionDefinitionFile":
        """Validate that function definitions are usable.

        Returns:
            The validated function definitions file.

        Raises:
            ValueError: If no functions were provided or names are duplicated.
        """
        validate_non_empty_list(self.functions, "functions file")

        names = [function.name for function in self.functions]
        if len(names) != len(set(names)):
            raise ValueError("function names must be unique")
        return self


class FunctionCallResult(BaseModel):
    """Represent a generated function call result.

    Attributes:
        prompt: Original natural-language request.
        name: Name of the selected function.
        parameters: Arguments selected for the function call.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]

    @model_validator(mode="after")
    def validate_function_call_result(self) -> "FunctionCallResult":
        """Validate that the generated function call result is complete.

        Returns:
            The validated function call result.

        Raises:
            ValueError: If prompt or function name is empty.
        """
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not self.name.strip():
            raise ValueError("function name must not be empty")
        return self


def validate_call_against_definition(
        call: FunctionCallResult,
        function: FunctionDefinition,
) -> None:
    """Validate a generated call against its function definition.

    Args:
        call: Generated function call result.
        function: Function definition selected for the call.

    Raises:
        ValueError: If the call name does not match the function definition, or
            if parameters are missing, extra, or have invalid types.
    """
    if call.name != function.name:
        raise ValueError("call name does not match function definition")

    expected_parameters = set(function.parameters.keys())
    received_parameters = set(call.parameters.keys())
    missing_parameters = expected_parameters - received_parameters
    if missing_parameters:
        raise ValueError(
            f"missing parameters: {', '.join(sorted(missing_parameters))}"
        )

    extra_parameters = received_parameters - expected_parameters
    if extra_parameters:
        raise ValueError(
            f"extra parameters: {', '.join(sorted(extra_parameters))}"
        )

    for parameter_name, parameter_definition in function.parameters.items():
        value = call.parameters[parameter_name]

        if (
                parameter_definition.type == "string"
                and not isinstance(value, str)
        ):
            raise ValueError(f"{parameter_name} must be a string")

        if parameter_definition.type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{parameter_name} must be a number")

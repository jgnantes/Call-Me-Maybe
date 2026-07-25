from typing import Any, Optional
from models import FunctionDefinition
import llm_sdk as llm
import functions as f


def get_input_ids(text: str, model: llm.Small_LLM_Model) -> list[int]:
    """Encode text into token IDs.

    Args:
        text: Text to encode.
        model: LLM wrapper used for tokenization.

    Returns:
        Token IDs representing the text.
    """
    return model.encode(text)[0].tolist()


def get_logits_from_text(
        text: str,
        model: llm.Small_LLM_Model,
) -> list[float]:
    """Get next-token logits for a text prompt.

    Args:
        text: Text prompt used as model context.
        model: LLM wrapper used for inference.

    Returns:
        Logit scores for every token in the model vocabulary.
    """
    return model.get_logits_from_input_ids(get_input_ids(text, model))


def choose_next_token_id(
        logits: list[float],
        allowed: Optional[list[int]] = None,
) -> int:
    """Choose the highest-scoring token ID.

    Args:
        logits: Logit scores indexed by token ID.
        allowed: Optional token IDs allowed for selection.

    Returns:
        Token ID with the highest score among the allowed tokens.
    """
    if allowed:
        return max(allowed, key=lambda token_id: logits[token_id])
    return max(range(len(logits)), key=logits.__getitem__)


def encode_allowed_texts(
        allowed_texts: list[str],
        model: llm.Small_LLM_Model,
) -> list[list[int]]:
    """Encode allowed output strings into token ID sequences.

    Args:
        allowed_texts: Candidate strings that the decoder may generate.
        model: LLM wrapper used for tokenization.

    Returns:
        Token ID sequences for each allowed string.
    """
    return [
        get_input_ids(text, model)
        for text in allowed_texts
    ]


def choose_from_allowed_texts(
        prompt: str,
        allowed_texts: list[str],
        model: llm.Small_LLM_Model,
) -> str:
    """Choose one allowed text using constrained greedy decoding.

    Args:
        prompt: Prompt used as model context.
        allowed_texts: Candidate strings that may be generated.
        model: LLM wrapper used for inference and tokenization.

    Returns:
        Selected allowed text.

    Raises:
        ValueError: If no allowed text is provided or decoding gets stuck.
    """
    if not allowed_texts:
        raise ValueError("allowed texts must not be empty")

    allowed_token_ids = encode_allowed_texts(allowed_texts, model)
    generated_tokens: list[int] = []
    prompt_tokens = get_input_ids(prompt, model)

    while generated_tokens not in allowed_token_ids:
        possible_next_tokens = []

        for token_ids in allowed_token_ids:
            if token_ids[:len(generated_tokens)] == generated_tokens:
                if len(token_ids) > len(generated_tokens):
                    possible_next_tokens.append(
                        token_ids[len(generated_tokens)]
                    )

        if not possible_next_tokens:
            raise ValueError("no valid next token available")

        logits = model.get_logits_from_input_ids(
            prompt_tokens + generated_tokens
        )
        next_token_id = choose_next_token_id(
            logits,
            allowed=possible_next_tokens,
        )
        generated_tokens.append(next_token_id)

    return model.decode(generated_tokens)


def build_function_selection_prompt(
        prompt: str,
        functions: list[FunctionDefinition],
) -> str:
    """Build a prompt for selecting the best function.

    Args:
        prompt: Natural-language request to classify.
        functions: Available function definitions.

    Returns:
        Prompt containing the request and available functions.
    """
    function_lines = [
        f"{function.name}: {function.description}"
        for function in functions
    ]

    return (
        "Select the best function for the user request.\n"
        "Available functions:\n"
        + "\n".join(function_lines)
        + f"\nUser request: {prompt}\n"
        "Function:"
    )


def choose_function(
        prompt: str,
        functions: list[FunctionDefinition],
        model: llm.Small_LLM_Model,
) -> FunctionDefinition:
    """Choose a function name for a prompt using constrained decoding.

    Args:
        prompt: Natural-language request to convert into a function call.
        functions: Available function definitions.
        model: LLM wrapper used for inference and tokenization.

    Returns:
        Selected function name.

    Raises:
        ValueError: If no function definitions are provided.
    """
    if not functions:
        raise ValueError("functions must not be empty")

    allowed_function_names = [
        f" {function.name}"
        for function in functions
    ]

    selection_prompt = build_function_selection_prompt(prompt, functions)
    selected_name = choose_from_allowed_texts(
        selection_prompt,
        allowed_function_names,
        model,
    ).strip()

    for function in functions:
        if function.name == selected_name:
            return function

    raise ValueError("selected function was not found")



if __name__ == "__main__":
    model = llm.Small_LLM_Model()
    string = "Tio Chico tinha um sítio"
    input_ids = model.encode(string)[0].tolist()

    for _ in range(55):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = choose_next_token_id(logits)
        input_ids.append(next_token_id)

    print("\n\nTESTE DE COMPLETAR TEXTO")
    print(f"\nANTES:  {string}")
    print(f"\nDEPOIS: {model.decode(input_ids)}")
    print()

    print("\n\nTESTE DE RESPONDER PERGUNTA")
    allowed_texts = [
        " yes",
        " no",
    ]
    allowed_token_ids = [
        model.encode(text)[0].tolist()[0]
        for text in allowed_texts
    ]

    question = "Is 2 plus 2 equal to 5?"
    logits = get_logits_from_text(question, model)
    best_allowed_token_id = max(
        allowed_token_ids,
        key=lambda token_id: logits[token_id],
    )
    print(question, model.decode([best_allowed_token_id]))
    for token_id in allowed_token_ids:
        print(model.decode([token_id]), logits[token_id])


    print("\n\nTESTE DE CHAMADA DE FUNÇÃO")
    from json_utils import load_function_definition_file

    functions_file = load_function_definition_file(
        "data/input/functions_definition.json"
    )

    prompt = "Lepetipotipola"
    print(f"Prompt: {prompt}")
    print("Function:", end=" ")
    print(
        choose_function(
            prompt,
            functions_file.functions,
            model,
        ).name,
    )

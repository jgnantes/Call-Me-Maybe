from typing import Any, Optional
import llm_sdk as llm
import functions as f


def get_logits_from_text(text: str, model: llm.Small_LLM_Model) -> list:
    """ """
    return model.get_logits_from_input_ids(model.encode(text)[0].tolist())


def choose_next_token_id(
        logits: list,
        allowed: Optional[list] = None
    ) -> int:
    """ """
    if allowed and allowed is not []:
        return max(allowed, key=lambda token_id: logits[token_id])
    return max(range(len(logits)), key=logits.__getitem__)


def choose_function(prompt: str, model: llm.Small_LLM_Model) -> dict[str, Any]:
    """ """
    allowed_function_names = [
        " fn_add_numbers",
        " fn_get_square_root",
        " fn_greet",
        " fn_reverse_string",
        " fn_substitute_string_with_regex",
    ]
    allowed_token_ids = [
        model.encode(name)[0].tolist()[0]
        for name in allowed_function_names
    ]
    for _ in range(5):
        logits = get_logits_from_text(prompt, model)
        next_token_id = choose_next_token_id(
            logits, allowed=allowed_token_ids)
        prompt += model.decode(next_token_id)
    return prompt



if __name__ == "__main__":
    model = llm.Small_LLM_Model()
    string = "Every Canadian is born"
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
    allowed_texts = [
        " fn_add_numbers",
        " fn_greet",
        " fn_reverse_string",
        " fn_get_square_root",
        " fn_substitute_string_with_regex",
    ]
    allowed_token_ids = [
        model.encode(text)[0].tolist()[0]
        for text in allowed_texts
    ]

    question = "What is the sum of 2 and 3?"
    logits = get_logits_from_text(question, model)
    best_allowed_token_id = max(
        allowed_token_ids,
        key=lambda token_id: logits[token_id],
    )
    print(choose_function(question, model))

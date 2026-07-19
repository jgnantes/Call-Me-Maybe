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


def choose_function(prompt: str, model: llm.Small_LLM_Model) -> str:
    """ """
    allowed_function_names = [
        " fn_add_numbers",
        " fn_get_square_root",
        " fn_greet",
        " fn_reverse_string",
        " fn_substitute_string_with_regex",
    ]
    allowed_token_ids = [
        model.encode(name)[0].tolist()
        for name in allowed_function_names
    ]

    generated_tokens: list[int] = []
    original_prompt = prompt

    while generated_tokens not in allowed_token_ids:
        possible_next_tokens = []

        for token_ids in allowed_token_ids:
            if token_ids[:len(generated_tokens)] == generated_tokens:
                if len(token_ids) > len(generated_tokens):
                    possible_next_tokens.append(token_ids[len(generated_tokens)])

        logits = model.get_logits_from_input_ids(
            model.encode(original_prompt)[0].tolist() + generated_tokens
        )

        next_token_id = choose_next_token_id(
            logits,
            allowed=possible_next_tokens,
        )
        generated_tokens.append(next_token_id)

    return model.decode(generated_tokens).strip()



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

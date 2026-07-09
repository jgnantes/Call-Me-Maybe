import llm_sdk as llm


if __name__=="__main__":
    model = llm.Small_LLM_Model()
    string = "Every Canadian is born"
    input_ids = model.encode(string)[0].tolist()

    for _ in range(25):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = logits.index(max(logits))
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
    logits = model.get_logits_from_input_ids(
        model.encode(question)[0].tolist()
    )
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
        model.encode(text)[0].tolist()
        for text in allowed_texts
    ]

    question = "What is the sum of 2 and 3?"
    logits = model.get_logits_from_input_ids(
        model.encode(question)[0].tolist()
    )
    best_allowed_token_id = max(
        allowed_token_ids,
        key=lambda token_id: logits[token_id],
    )

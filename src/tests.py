import llm_sdk as llm


if __name__=="__main__":
    model = llm.Small_LLM_Model()
    input_ids = model.encode("Atirei um pau no gato-to")[0].tolist()

    for _ in range(25):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = logits.index(max(logits))
        input_ids.append(next_token_id)

    print(model.decode(input_ids))

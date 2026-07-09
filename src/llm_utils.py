import llm_sdk as llm


def choose_next_token_id(logit_indexes: list) -> int:
    """ """
    return max(range(len(logit_indexes)), key=logit_indexes.__getitem__)


def get_logits_from_text(text: str, model: llm.Small_LLM_Model) -> list:
    """ """
    return model.get_logits_from_input_ids(model.encode(text)[0].tolist())
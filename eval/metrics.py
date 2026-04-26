def get_bias_score(model, prompt):
    tokens = model.to_tokens(prompt)
    logits = model(tokens)

    he = model.to_single_token(" he")
    she = model.to_single_token(" she")

    last = logits[0, -1]
    return (last[he] - last[she]).item()
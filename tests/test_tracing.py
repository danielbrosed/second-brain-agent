from second_brain_agent import tracing


def test_mask_redacts_credentials_anywhere_in_a_payload():
    key = "sk_live_" + "Z9y8" * 6
    data = {"answer": f"The key is {key}.", "sources": [f"note with {key}"], "attempts": 2}

    masked = tracing.mask(data=data)

    assert key not in str(masked)
    assert masked["answer"] == "The key is [REDACTED]."
    assert masked["attempts"] == 2


def test_mask_keeps_where_a_key_lives():
    text = '1Password "Tidepool" → "Stripe live", and the env var STRIPE_SECRET_KEY'
    assert tracing.mask(data=text) == text


def test_untraced_root_is_a_no_op():
    with tracing.root("answer-question", input={"question": "q"}) as observation:
        observation.update(output={"answer": "a"})
    assert observation.trace_id is None
    assert tracing.handler() is None
    assert tracing.trace_id_for("capture-123") is None

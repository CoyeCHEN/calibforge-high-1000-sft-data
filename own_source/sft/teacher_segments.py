"""Actual-request Qwen projection, with each agent decision supervised once.

The official Qwen template, also required at serving, owns historical thinking
projection. Segments may overlap as input; only newly assigned targets get loss.
No observations are reconstructed from the larger raw tool results.
"""

from dataclasses import dataclass

from .qwen35 import digest, normalize_message, validate_pairs


@dataclass
class Decision:
    call_id: str
    row: dict
    evidence: dict


def project_decision(tokenizer, call, *, enable_thinking=True, projection_source="provider_payload"):
    """Verify the exact native generation prefix used by the student."""
    if projection_source not in ("provider_payload", "published_messages"):
        raise ValueError("Unknown conditioning evidence contract")
    if call.get("context_fidelity") != projection_source:
        raise ValueError("Conditioning evidence does not match the required source contract")
    payload = call["provider_payload"]
    history = [normalize_message(x) for x in payload["messages"]]
    response = normalize_message(call["assistant_response"])
    if response["role"] != "assistant":
        raise ValueError("Target must be an actual assistant response")
    if any(
        marker in response.get("reasoning_content", "")
        for marker in ("<think>", "</think>")
    ):
        raise ValueError("Embedded thinking delimiter needs explicit semantic review")
    messages = history + [response]
    linkage, pending = validate_pairs(messages)
    if not enable_thinking and response.get("reasoning_content"):
        raise ValueError("Thinking-off targets must not contain separate reasoning")
    options = dict(tools=payload.get("tools"), enable_thinking=enable_thinking)
    prefix = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True, **options
    )
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **options
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    native_prefix = tokenizer.apply_chat_template(
        history, tokenize=True, return_dict=False, add_generation_prompt=True, **options
    )
    if prefix_ids != native_prefix or not rendered.startswith(prefix):
        raise ValueError("DEPLOYMENT_PREFIX_MISMATCH:" + call["call_id"])
    opener = ("<|im_start|>assistant\n<think>\n" if enable_thinking else
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    if not prefix.endswith(opener):
        raise ValueError("Official generation prefix changed")
    # Empty reasoning produces two adjacent newlines across the generation
    # boundary. A one-pass BPE merges them and changes an already supplied input
    # token. Preserve the deployed prefix and tokenize the generated suffix.
    suffix = rendered[len(prefix) :]
    suffix_encoding = tokenizer(
        suffix, add_special_tokens=False, return_offsets_mapping=True
    )
    suffix_ids = suffix_encoding["input_ids"]
    ids = prefix_ids + suffix_ids
    canonical_ids = tokenizer.encode(rendered, add_special_tokens=False)
    decoded_prefix = tokenizer.decode(
        prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    # The pinned tokenizer applies NFC to historical input at serving too.
    # Its normal input normalization must not be confused with a changed
    # generated target. Targets still have to round-trip byte-for-byte.
    normalized_prefix = tokenizer.backend_tokenizer.normalizer.normalize_str(prefix)
    if (
        decoded_prefix != normalized_prefix
        or tokenizer.decode(
            suffix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        != suffix
        or tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        != decoded_prefix + suffix
    ):
        raise ValueError("Generation-boundary tokenization changed template text")
    left = len(prefix_ids)
    if left >= len(ids):
        raise ValueError("Empty assistant target")
    eos = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    separator = tokenizer.encode("\n", add_special_tokens=False)
    if not eos or ids[-len(eos) - len(separator) :] != eos + separator:
        raise ValueError("Official message termination changed")
    right = len(ids) - len(separator)
    row = dict(
        input_ids=ids,
        attention_mask=[1] * len(ids),
        labels=[-100] * left + ids[left:right] + [-100] * len(separator),
        images=[],
        videos=[],
        audios=[],
    )
    thinking = response.get("reasoning_content", "").strip()
    body_start = len(thinking) + len("\n</think>\n\n") if enable_thinking else 0
    body_end = body_start + len(response["content"].strip())
    ending_start = suffix.rfind("<|im_end|>")
    classes = {
        k: 0 for k in ["thinking", "thinking_boundary", "body", "action", "ending"]
    }
    for start, end in suffix_encoding["offset_mapping"][: -len(separator)]:
        category = (
            "thinking"
            if start < len(thinking)
            else "thinking_boundary"
            if start < body_start
            else "body"
            if start < body_end
            else "action"
            if start < ending_start
            else "ending"
        )
        classes[category] += 1
    if sum(classes.values()) != right - left:
        raise AssertionError("Unclassified supervision token")
    evidence = dict(
        call_id=call["call_id"],
        context_fidelity=projection_source,
        input_tokens=len(ids),
        prefix_tokens=left,
        supervised_range=[left, right],
        supervised_tokens=right - left,
        generation_prefix_sha256=digest(prefix_ids),
        input_sha256=digest(ids),
        labels_sha256=digest(row["labels"]),
        provider_payload_sha256=digest(payload),
        source_history_sha256=digest(payload["messages"]),
        normalized_history_sha256=digest(history),
        tools_sha256=digest(payload.get("tools")),
        linkage=linkage,
        pending_target_tool_ids=pending,
        exact_deployment_prefix=True,
        thinking_opener_policy=("Official serving prefix supplies the thinking opener; input only" if enable_thinking else
                               "Official serving prefix supplies empty closed thinking; input only"),
        enable_thinking=enable_thinking,
        termination_policy="Supervise im_end EOS; official post-EOS message-separator newline is input-only",
        post_eos_separator_tokens=len(separator),
        historical_thinking_policy="Pinned official Jinja last-real-user policy; same renderer required at serving",
        tokenization_policy="Exact native generation-prefix IDs followed by encoded rendered target suffix",
        input_prefix_normalized_by_official_tokenizer=decoded_prefix != prefix,
        target_text_roundtrip_exact=True,
        canonical_full_render_tokens=len(canonical_ids),
        generation_boundary_changes_bpe=ids != canonical_ids,
        supervision_token_classes=classes,
        classification_policy="Token assigned by starting character offset in official rendered target",
        target_tool_names=[
            t["function"]["name"] for t in response.get("tool_calls", [])
        ],
        target_has_finish=any(
            t["function"]["name"] == "finish" for t in response.get("tool_calls", [])
        ),
        reasoning_projection=("DeepSeek reasoning_content -> Qwen official reasoning_content; provider signatures omitted"
                              if projection_source == "provider_payload" else
                              "Published assistant content retained as ordinary body; no fabricated hidden thinking"),
        observation_basis=("Actual recorded provider payload, including native toolResultBudget previews"
                           if projection_source == "provider_payload" else
                           "Published ChatML tool observations, unmodified; teacher HTTP payload was not published"),
        truncation=False,
    )
    return Decision(call["call_id"], row, evidence)


def make_segments(decisions, *, cutoff, processor_tokens=0):
    """Greedy maximal compatible consecutive groups; never fabricate a prefix."""
    if cutoff <= 0 or processor_tokens < 0:
        raise ValueError("Positive cutoff required")
    identities = [d.call_id for d in decisions]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Unique nonempty call identities required")
    groups = []
    current = []
    quarantine = []
    for decision in decisions:
        ids = decision.row["input_ids"]
        if len(ids) + processor_tokens > cutoff:
            if current:
                groups.append(current)
                current = []
            quarantine.append(
                dict(
                    call_id=decision.call_id,
                    reason="INPUT_PLUS_TARGET_EXCEEDS_CUTOFF",
                    tokens=len(ids),
                    processor_tokens=processor_tokens,
                    cutoff=cutoff,
                    input_sha256=decision.evidence["input_sha256"],
                )
            )
            continue
        compatible = all(
            ids[: len(old.row["input_ids"])] == old.row["input_ids"] for old in current
        )
        if current and not compatible:
            groups.append(current)
            current = []
        current.append(decision)
    if current:
        groups.append(current)
    samples = []
    assigned = []
    for number, group in enumerate(groups):
        last = group[-1]
        ids = last.row["input_ids"]
        labels = [-100] * len(ids)
        mapping = []
        for decision in group:
            left, right = decision.evidence["supervised_range"]
            if ids[: len(decision.row["input_ids"])] != decision.row["input_ids"]:
                raise AssertionError("A target acquired a history it never saw")
            if any(x != -100 for x in labels[left:right]):
                raise AssertionError("Assistant target ranges overlap")
            labels[left:right] = ids[left:right]
            assigned.append(decision.call_id)
            mapping.append(
                dict(
                    call_id=decision.call_id,
                    supervised_range=[left, right],
                    generation_prefix_sha256=decision.evidence[
                        "generation_prefix_sha256"
                    ],
                    original_input_sha256=decision.evidence["input_sha256"],
                )
            )
        row = dict(
            input_ids=ids,
            attention_mask=[1] * len(ids),
            labels=labels,
            images=[],
            videos=[],
            audios=[],
        )
        samples.append(
            dict(
                row=row,
                evidence=dict(
                    segment_index=number,
                    cutoff=cutoff,
                    input_tokens=len(ids),
                    supervised_tokens=sum(x != -100 for x in labels),
                    processor_tokens=processor_tokens,
                    input_sha256=digest(ids),
                    labels_sha256=digest(labels),
                    decision_mapping=mapping,
                    overlap_history_input_only=True,
                    all_target_prefixes_exact=True,
                    truncation=False,
                ),
            )
        )
    expected = [x for x in identities if x not in {q["call_id"] for q in quarantine}]
    if assigned != expected:
        raise AssertionError("Targets omitted, duplicated, or reordered")
    return dict(
        samples=samples,
        quarantine=quarantine,
        target_count=len(assigned),
        all_targets_exactly_once=not quarantine,
        original_target_count=len(identities),
    )

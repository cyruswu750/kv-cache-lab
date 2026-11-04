"""Prompt fixtures.

- BOUNDARY_TEXT: tokenized, then cycled/truncated to an exact token count, so
  boundary cases have controlled lengths (255, 256, 257, 511, 512, 513).
- NATURAL_PROMPTS: chat-templated prompts of natural lengths.
- CALIBRATION_TEXT: used only by scripts/m0_calibrate.py to set tolerances
  before acceptance cases are evaluated; never used by acceptance cases.
"""

BOUNDARY_TEXT = (
    "The lighthouse keeper climbed the spiral stairs every evening at dusk, counting the steps out of habit "
    "even though he had known the number for thirty years. At the top he trimmed the wick, polished the great "
    "lens with a soft cloth, and checked the clockwork that turned the beam across the water. Ships no longer "
    "depended on him the way they once had; satellites and radio beacons did most of the work now. Still, the "
    "harbor authority kept the light running, partly from tradition and partly because instruments fail and a "
    "visible flash on the horizon is hard to misunderstand. On clear nights he could see the lights of the "
    "fishing fleet strung out like beads, and he liked to guess which boat belonged to which family. On foggy "
    "nights he started the horn, a low two-tone groan that rattled the windows of the cottages below. In the "
    "morning he wrote a short entry in the logbook: the weather, the visibility, any vessels that had passed "
    "close, and whether the mechanism had needed attention. Most entries were dull. A few were not: the night "
    "a sailboat lost its mast in a squall, the winter the bay froze for a week, the evening a pod of whales "
    "surfaced so close to the rocks that he could hear them breathe. He kept the old logbooks on a shelf in the "
    "kitchen, and sometimes visitors asked to read them. He always said yes, and he always warned them that "
    "the interesting parts were few and far between, which was exactly how a lighthouse ought to be. "
)

_PASSAGE_STORAGE = (
    "A key-value store keeps data as pairs of keys and values and offers operations such as get, put, and "
    "delete. Many modern stores use a log-structured merge tree: writes go first to an in-memory table and a "
    "write-ahead log, and when the memory table fills up it is flushed to disk as an immutable sorted file. "
    "Background compaction later merges these files, discarding overwritten and deleted entries. Reads may "
    "need to consult several files, so stores keep Bloom filters and block indexes to skip files that cannot "
    "contain a key. The design trades some read amplification and background work for fast sequential writes."
)

_PASSAGE_TRANSFORMER = (
    "During autoregressive generation a transformer produces one token at a time. Each attention layer "
    "computes key and value vectors for every token it has seen. Recomputing them for the whole sequence at "
    "every step would be wasteful, so inference engines store them in a key-value cache and only compute the "
    "new token's projections. The cache grows linearly with sequence length and with the number of layers and "
    "attention heads, which is why long contexts consume a large share of accelerator memory. Systems that "
    "serve many requests with shared prefixes, such as a common system prompt, can reuse cached keys and "
    "values across requests instead of recomputing the shared prefix, provided the positions and masks match."
)

NATURAL_PROMPTS = {
    "nat_short": [
        {"role": "user", "content": "In one sentence, what does a lighthouse do?"},
    ],
    "nat_medium": [
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": _PASSAGE_STORAGE + "\n\nExplain the main trade-off described above in two "
                                    "sentences, then give one example workload where it is a good fit."},
    ],
    "nat_long": [
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": _PASSAGE_STORAGE + "\n\n" + _PASSAGE_TRANSFORMER + "\n\n" + _PASSAGE_STORAGE
                                    + "\n\nCompare the two systems described above. Which one reuses previously "
                                    "computed results, and what must match for that reuse to be correct?"},
    ],
    # Added after run1 so that one natural prompt exceeds 512 tokens (partial restore).
    "nat_xlong": [
        {"role": "system", "content": "You are a careful reading assistant."},
        {"role": "user", "content": "Story:\n" + BOUNDARY_TEXT + "\n\nNotes:\n" + _PASSAGE_STORAGE + "\n\n"
                                    + _PASSAGE_TRANSFORMER + "\n\nIn three bullet points, say what the story's "
                                    "logbook and the two notes have in common."},
    ],
}

CALIBRATION_TEXT = (
    "Tide pools form where the sea retreats twice a day and leaves water trapped among the rocks. The animals "
    "that live there must tolerate sudden changes in temperature, salinity, and oxygen. Barnacles close their "
    "shells and wait; anemones fold their tentacles inward; small crabs hide beneath stones until the water "
    "returns. Biologists divide the shore into zones by how long each band stays submerged, and the species "
    "found in each zone differ sharply even across a few meters of height. Storms rearrange the boulders and "
    "reset the competition for space, so a pool that was crowded with mussels one year may be dominated by "
    "seaweed the next. Visitors are asked to step only on bare rock, to return any stone they lift to its "
    "original position, and to leave the animals where they found them. A careful observer who sits still for "
    "ten minutes will usually see far more than one who hurries from pool to pool. "
)


def exact_length_ids(tokenizer, text: str, n: int) -> list[int]:
    """Token ids of `text`, cycled and truncated to exactly n tokens."""
    base = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not base:
        raise ValueError("fixture text tokenized to nothing")
    reps = -(-n // len(base))
    return (base * reps)[:n]


def chat_ids(tokenizer, messages) -> list[int]:
    # transformers 5.22.1 returns a BatchEncoding here (verified), not a bare list.
    enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return list(enc["input_ids"])

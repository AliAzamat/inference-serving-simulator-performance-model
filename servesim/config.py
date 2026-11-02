"""Cost model and limits for the simulated server. Every time is in seconds."""

PREFILL_S_PER_TOKEN = 0.00005       # reading one prompt token
DECODE_BASE_S = 0.012               # one decode step, however small the batch
DECODE_S_PER_KV_TOKEN = 0.0000002   # extra time per cached token read in a step
KV_CAPACITY_TOKENS = 60000          # how many tokens of KV cache fit in memory
MAX_BATCH = 64                      # most requests the server will run together


def prefill_time(prompt_tokens):
    return prompt_tokens * PREFILL_S_PER_TOKEN


def decode_step_time(total_kv_tokens):
    return DECODE_BASE_S + DECODE_S_PER_KV_TOKEN * total_kv_tokens

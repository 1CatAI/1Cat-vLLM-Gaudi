# DeepSeek V4.1 unified full-acceleration E2E validation

- Date: 2026-09-21
- Source: `5fa3e083` plus the V2 B2 completion ownership repair in this worktree
- Hardware: 4 x Gaudi2, TP2 x PP2, DSpark off
- Service: max model length 1,048,576; max batched tokens 8,192; max sequences 32; KV block size 128; 8,193 blocks
- Public base URL: `http://dx.1catai.com:54868/v1`
- Model: `DeepSeek-V4.1-Flash`

## Performance

2048 input tokens, 256 generated tokens, greedy, three rounds; first 10 ITLs discarded per round.

- Round 1: 12.276568 ms/token (81.456 tokens/s)
- Round 2: 12.300522 ms/token (81.297 tokens/s)
- Round 3: 12.301963 ms/token (81.288 tokens/s)
- Mean: 12.293018 ms/token (81.347 tokens/s)
- Post-repair B1 confirmation: 12.307399 ms/token (81.252 tokens/s)

## Functional and lifecycle checks

- 8/8 generation smoke cases passed with natural EOS.
- Long generation: 1,237 completion tokens / 2,002 Chinese characters, natural EOS, no repeated paragraph or invalid Unicode.
- B2 concurrent requests: 2/2 passed after disabling the single-request asynchronous continuation inside multi-request scheduler steps.
- Streaming cancellation followed by recovery request: passed; recovery answer was `63`.
- Public gateway: `/v1/models`, non-streaming, streaming with `[DONE]`, and B2 concurrent requests passed.
- Backend health after all tests: HTTP 200.

## Fixes made during unified validation

1. V2 asynchronous completion is selected only for a scheduler step containing one request. B2+ uses the batch-generic synchronous completion contract, preventing PP packet and request-generation ownership collisions.
2. The gateway now targets port 18422, admits up to 32 requests, and emits ASCII-safe SSE JSON so U+2028/U+2029 cannot split frames in line-oriented clients. Client disconnects are handled without a gateway traceback.

# Using OpenRouter

Sellbird can think through any OpenAI-compatible chat endpoint instead of the Claude or Codex CLI. That endpoint can be [OpenRouter](https://openrouter.ai) itself, including its free `:free` models, or a gateway that speaks the same API.

Unlike Claude and Codex, this is billed per call by whoever runs the endpoint. On free models it costs nothing.

## Set it up

1. Get an API key. For OpenRouter, that's openrouter.ai > Keys; for a gateway, ask whoever runs it.
2. Put the key in `.env`:

   ```
   OPENROUTER_API_KEY=your-key
   ```

3. Choose the provider in `sellbird.yaml` (or `sellbird.local.yaml`):

   ```yaml
   brain:
     provider: openrouter
     model: meta-llama/llama-3.3-70b-instruct:free   # any model id the endpoint accepts
     # base_url: https://openrouter.ai/api/v1        # the default
     # api_key_env: OPENROUTER_API_KEY               # the env variable holding the key
   ```

   To use a gateway instead, set `base_url` to the gateway's OpenAI-compatible URL (the part before `/chat/completions`). Set `model` to a model id or alias it accepts. For example:

   ```yaml
   brain:
     provider: openrouter
     base_url: https://gateway.example.com/api/v1
     model: free
   ```

4. Run `sellbird status --check`. It reads the endpoint's model list to confirm the key and URL work. No prompt is sent.

## How it behaves

- **`model` is required.** No model is picked for you.
- **Bad settings pause Sellbird.** A missing key, a refused key (401/403) or a wrong URL or model (404) pauses Sellbird until you change the setting. The dashboard and `sellbird status` show why.
- **Rate and credit limits pause it for an hour.** A rate limit or empty credit (429/402) pauses calls for an hour. Server errors and timeouts are retried.
- **There is no quota gauge.** The daily budget uses Sellbird's own call counter (`max_daily_calls`). OpenRouter calls are counted separately from Claude and Codex calls.
- **The key stays private.** It is only read from the environment and is never logged or shown.
- **Free models write less well.** They are fine for scoring and research. Check their drafts before turning on autopilot.

"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

# README documents the project-root .env file; local lab setups also place it in .venv.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / ".venv" / ".env")

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.1-flash-lite"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "mock-key"

    client = OpenAI(api_key=api_key)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "mock-key"
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time

    meta = response.usage_metadata
    usage = {
        "input_tokens": meta.prompt_token_count or 0,
        "output_tokens": meta.candidates_token_count or 0,
    }
    return response.text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY") or "mock-key"
    client = anthropic.Anthropic(api_key=api_key)

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.
    """

    def _cost(model_key: str, usage: dict) -> float:
        rates = PRICING_1M_TOKENS[model_key]
        return (
            usage["input_tokens"] * rates["input"]
            + usage["output_tokens"] * rates["output"]
        ) / 1_000_000

    # Call GPT-4o
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_cost = _cost("gpt-4o", gpt4o_usage)

    # Call GPT-4o-mini
    mini_text, mini_lat, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    mini_cost = _cost("gpt-4o-mini", mini_usage)

    # Call Gemini 2.5 Flash
    gem_text, gem_lat, gem_usage = call_gemini(prompt, model=GEMINI_MODEL)
    gem_cost = _cost("gemini-2.5-flash", gem_usage)

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": mini_cost,
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gem_text,
            "latency": gem_lat,
            "cost": gem_cost,
            "input_tokens": gem_usage["input_tokens"],
            "output_tokens": gem_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.
    Maintains the last 3 turns of conversation history for context.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    dummy_mode = not api_key
    if dummy_mode:
        print("\033[93m[System Warning] GEMINI_API_KEY environment variable not set. Running in offline demo mode.\033[0m")
    else:
        api_key = api_key

    print("\n\033[94m================================================")
    print("Vin Smart Future - Intelligent Chat Assistant")
    print("Powered by Google Gemini")
    print("Type 'quit' or 'exit' to end the session.")
    print("================================================\033[0m\n")

    client = None if dummy_mode else genai.Client(api_key=api_key)
    history = []  # list of types.Content; 3 turns = 6 messages (user + model)

    candidate_models = list(dict.fromkeys([
        GEMINI_MODEL,
        "gemini-3.1-flash-lite-preview",
        "gemini-3.8-flash",
        "gemini-2.5-flash-lite",
    ]))

    while True:
        try:
            user_input = input("\033[92mYou: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        contents = history + [
            types.Content(role="user", parts=[types.Part(text=user_input)])
        ]

        print("\033[96mGemini: \033[0m", end="", flush=True)
        reply = ""

        if dummy_mode:
            reply = "Demo mode: please configure GEMINI_API_KEY in .venv/.env to enable Gemini responses."
            print(reply + "\n")
        else:
            last_error = None
            for model_name in candidate_models:
                for attempt in range(3):
                    try:
                        for chunk in client.models.generate_content_stream(
                            model=model_name,
                            contents=contents,
                        ):
                            if chunk.text:
                                print(chunk.text, end="", flush=True)
                                reply += chunk.text
                        print("\n")
                        break
                    except Exception as exc:
                        last_error = exc
                        error_text = str(exc)
                        if "503" not in error_text and "429" not in error_text and "UNAVAILABLE" not in error_text:
                            break
                        if attempt < 2:
                            delay = 1 * (2 ** attempt)
                            print(f"\n[Temporary API issue; retrying in {delay}s...]", end="", flush=True)
                            time.sleep(delay)
                if reply:
                    break

            if not reply:
                reply = (
                    "Gemini is temporarily unavailable (API status 503 or model capacity). "
                    "Your input is kept in the chat history, so you can retry shortly."
                )
                print(f"\n\033[93m{reply}\033[0m\n")

        contents.append(types.Content(role="model", parts=[types.Part(text=reply)]))
        history = contents[-6:]  # keep only the last 3 turns


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(base_delay * (2 ** attempt))
    raise last_error


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.
    Each result gets an extra key "prompt" with the original prompt string.
    """
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.
    Columns: | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---:|---:|---:|",
    ]
    model_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }
    for result in results:
        prompt = str(result.get("prompt", "")).replace("|", "\\|").replace("\n", " ")
        for model_key, stats in result.items():
            if model_key not in model_names:
                continue
            response = str(stats["response"]).replace("|", "\\|").replace("\n", " ")
            response = response[:47] + "..." if len(response) > 50 else response
            lines.append(
                f"| {prompt} | {model_names[model_key]} | {response} | "
                f"{stats['latency']:.2f}s | {stats['input_tokens']}/{stats['output_tokens']} | "
                f"${stats['cost']:.6f} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
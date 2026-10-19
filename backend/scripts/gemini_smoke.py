"""Manual smoke test: one real Gemini request through LLMProvider, with no business tools.

Uses real Gemini quota, so it is not part of the test suite and never runs automatically.
Needs GEMINI_API_KEY in the environment or in backend/.env or the repository's .env.
Run from backend/:  python -m scripts.gemini_smoke
Prints only the assistant's reply (or a ProviderError, which never contains the key).
"""

import sys

from app.agent.llm import LLMProvider, Message, ProviderError
from app.agent.providers.gemini import GeminiProvider, GeminiSettings
from app.env_file import load_env_files

SYSTEM = "אתה עוזר ידידותי. ענה בעברית, במשפט אחד קצר."
QUESTION = "שלום! במשפט אחד: מה שלומך היום?"


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # a console without Hebrew/emoji support shows '?' instead of crashing
    load_env_files()
    try:
        provider: LLMProvider = GeminiProvider(GeminiSettings.from_env())
        reply = provider.complete(SYSTEM, [Message.user(QUESTION)], tools=[])
    except ProviderError as e:
        print(f"Gemini smoke test failed: {e}", file=sys.stderr)
        return 1
    print(reply.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

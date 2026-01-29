import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from google.colab import userdata  # type: ignore
except ImportError:  # Not running inside Colab
    userdata = None

load_dotenv()


def configure_openrouter(
    secret_name: str = "openrouter_API",
    env_var: str = "openrouter_API",
    session_var: str = "OPENROUTER_API_KEY"
) -> OpenAI:
    """Load the OpenRouter API key and create configured client.

    Args:
        secret_name: Name of the Colab secret to check.
        env_var: Name of the environment variable to check.
        session_var: Name to store the key in os.environ for session use.

    Returns:
        Configured OpenAI client pointing to OpenRouter API.
    """
    key = None

    if userdata is not None:
        try:
            key = userdata.get(secret_name)
            if key:
                print(f"Loaded OpenRouter key from Colab secret '{secret_name}'.")
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"Could not read Colab secret '{secret_name}': {exc}")

    if not key:
        key = os.getenv(env_var)
        if key:
            print(f"Loaded OpenRouter key from environment variable '{env_var}'.")

    if not key:
        raise RuntimeError(
            "OpenRouter API key not found. In Colab, add it via Settings -> Secrets as 'openrouter_API'. "
            "Locally, set openrouter_API in a .env file or export it before running this cell."
        )

    os.environ[session_var] = key
    print("OPENROUTER_API_KEY is configured for this session.")

    # Create and return the configured OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        default_headers={
            "HTTP-Referer": "https://github.com/smartaces/opencite",
            "X-Title": "OpenCite Bulk Loader"
        }
    )

    return client


# Run configuration and create global client
client = configure_openrouter()

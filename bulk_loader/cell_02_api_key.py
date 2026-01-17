import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from google.colab import userdata  # type: ignore
except ImportError:  # Not running inside Colab
    userdata = None

load_dotenv()


def configure_openai(secret_name: str = "openai_key", env_var: str = "OPENAI_API_KEY") -> None:
    """Load the OpenAI API key from Colab secrets, .env, or environment vars."""
    key = None

    if userdata is not None:
        try:
            key = userdata.get(secret_name)
            if key:
                print(f"✅ Loaded OpenAI key from Colab secret '{secret_name}'.")
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"⚠️ Could not read Colab secret '{secret_name}': {exc}")

    if not key:
        key = os.getenv(env_var)
        if key:
            print(f"✅ Loaded OpenAI key from environment variable '{env_var}'.")

    if not key:
        raise RuntimeError(
            "OpenAI API key not found. In Colab, add it via Settings → Secrets as 'openai_key'. "
            "Locally, set OPENAI_API_KEY in a .env file or export it before running this cell."
        )

    os.environ[env_var] = key
    print("🔐 OPENAI_API_KEY is configured for this session.")


configure_openai()

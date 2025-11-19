def format_location(country: str = "US", city: str = "New York", region: str = "New York") -> dict:
    """Return a location dict compatible with OpenAI web search requests."""
    return {
        "country": country,
        "city": city,
        "region": region
    }


print("✅ format_location helper ready (use it when passing user_location to search calls).")

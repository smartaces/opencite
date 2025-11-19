from typing import Any, Dict, List, Optional

from openai import OpenAI

client = OpenAI()


class OpenAISearchAgent:
    """OpenAI Web Search Agent using the Responses API."""

    def __init__(self, client: OpenAI, model: str = "gpt-5"):
        self.client = client
        self.model = model
        self.conversation_history: List[Dict[str, Any]] = []
        self.last_response_id: Optional[str] = None  # For GPT-5 reasoning continuity

    def search(
        self,
        query: str,
        search_context_size: str = "medium",
        user_location: Optional[Dict[str, str]] = None,
        force_search: bool = False,
        reasoning_effort: str = "low",
        verbosity: str = "medium",
        use_previous_reasoning: bool = True,
    ) -> Dict[str, Any]:
        tools: List[Dict[str, Any]] = []
        web_search_config: Dict[str, Any] = {"type": "web_search_preview"}

        if self.model not in ["o3", "o3-pro", "o4-mini"]:
            web_search_config["search_context_size"] = search_context_size

        if user_location:
            web_search_config["user_location"] = {
                "type": "approximate",
                **user_location,
            }

        tools.append(web_search_config)

        request_params: Dict[str, Any] = {
            "model": self.model,
            "tools": tools,
            "input": query,
        }

        if self.model.startswith("gpt-5"):
            if reasoning_effort == "minimal":
                reasoning_effort = "low"

            request_params["reasoning"] = {"effort": reasoning_effort}
            request_params["text"] = {"verbosity": verbosity}

            if use_previous_reasoning and self.last_response_id:
                request_params["previous_response_id"] = self.last_response_id

        if force_search:
            request_params["tool_choice"] = {"type": "web_search_preview"}

        try:
            response = self.client.responses.create(**request_params)
            if hasattr(response, "id"):
                self.last_response_id = response.id
            self.conversation_history.append({"query": query, "response": response})
            return response
        except Exception as exc:  # pragma: no cover - API/network errors
            return {"error": str(exc)}

    def extract_text_response(self, response: Any) -> str:
        try:
            if isinstance(response, dict) and "error" in response:
                return f"Error: {response['error']}"
            if hasattr(response, "output_text"):
                return response.output_text
            if hasattr(response, "output"):
                for output in response.output:
                    if getattr(output, "type", None) == "message" and hasattr(output, "content"):
                        for item in output.content:
                            if getattr(item, "type", None) == "output_text":
                                return getattr(item, "text", "")
                    elif isinstance(output, dict) and output.get("type") == "message":
                        for item in output.get("content", []):
                            if item.get("type") == "output_text":
                                return item.get("text", "")
            return "No text response found"
        except Exception as exc:
            return f"Error extracting response: {exc}"

    def extract_citations(self, response: Any) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        try:
            if hasattr(response, "output"):
                for output in response.output:
                    if getattr(output, "type", None) == "message" and hasattr(output, "content"):
                        for item in output.content:
                            if getattr(item, "type", None) == "output_text":
                                annotations = getattr(item, "annotations", [])
                                for ann in annotations:
                                    if getattr(ann, "type", None) == "url_citation":
                                        citations.append({
                                            "url": getattr(ann, "url", ""),
                                            "title": getattr(ann, "title", ""),
                                        })
                    elif isinstance(output, dict) and output.get("type") == "message":
                        for item in output.get("content", []):
                            if item.get("type") == "output_text":
                                for ann in item.get("annotations", []):
                                    if ann.get("type") == "url_citation":
                                        citations.append({
                                            "url": ann.get("url", ""),
                                            "title": ann.get("title", ""),
                                        })
        except Exception:
            pass
        return citations

    def get_search_actions(self, response: Any) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        try:
            if hasattr(response, "output"):
                for output in response.output:
                    if getattr(output, "type", None) == "web_search_call":
                        actions.append({
                            "id": getattr(output, "id", ""),
                            "status": getattr(output, "status", ""),
                            "action": getattr(output, "action", {}),
                        })
                    elif isinstance(output, dict) and output.get("type") == "web_search_call":
                        actions.append({
                            "id": output.get("id", ""),
                            "status": output.get("status", ""),
                            "action": output.get("action", {}),
                        })
        except Exception:
            pass
        return actions

    def clear_history(self) -> None:
        self.conversation_history = []
        self.last_response_id = None
        print("🗑️ Conversation history cleared")


search_agent = OpenAISearchAgent(client, model="gpt-5")
print(f"🤖 Search Agent initialized with {search_agent.model}!")
print("💡 Tip: GPT-5 reasoning chains persist across turns when previous_response_id is set.")


def print_search_result(response, show_citations: bool = True, show_actions: bool = False) -> None:
    """Pretty-print a response from the search agent."""
    text = search_agent.extract_text_response(response)

    print("=" * 80)
    print("📝 RESPONSE:")
    print("=" * 80)
    print(text)
    print()

    if show_citations:
        citations = search_agent.extract_citations(response)
        if citations:
            print("=" * 80)
            print("🔗 CITATIONS:")
            print("=" * 80)
            for i, cite in enumerate(citations, 1):
                print(f"{i}. {cite.get('title', 'Untitled')}")
                print(f"   URL: {cite.get('url', 'Unknown')}")
                print()

    if show_actions:
        actions = search_agent.get_search_actions(response)
        if actions:
            print("=" * 80)
            print("🔍 SEARCH ACTIONS:")
            print("=" * 80)
            for action in actions:
                print(f"ID: {action.get('id', '')}")
                print(f"Status: {action.get('status', '')}")
                if action.get('action'):
                    print(f"Action details: {action['action']}")
                print()

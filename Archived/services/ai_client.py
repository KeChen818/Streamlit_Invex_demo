from openai import AzureOpenAI

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except Exception:
    DefaultAzureCredential = None
    get_bearer_token_provider = None


def get_client(cfg: dict):
    """
    Supports either:
    1. Azure AD auth through DefaultAzureCredential
    2. API key auth
    """

    if DefaultAzureCredential and get_bearer_token_provider and not cfg.get("api_key"):
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )

        return AzureOpenAI(
            azure_endpoint=cfg["endpoint"],
            api_version=cfg["api_version"],
            azure_ad_token_provider=token_provider,
        )

    return AzureOpenAI(
        azure_endpoint=cfg["endpoint"],
        api_version=cfg["api_version"],
        api_key=cfg["api_key"],
    )


def build_base_system() -> str:
    return """
You are a risk inventory assistant for an enterprise risk management application.

STRICT RULES:
- Use only the provided DATA_CONTEXT_JSON.
- Do not invent risks, ratings, rationales, or management actions.
- Always cite group_id when discussing a specific risk.
- If the user asks about material risks, do not include non-material risks unless requested.
- If the data is insufficient, clearly state what is missing.
- Keep the response concise, structured, and suitable for risk management users.
"""


def ask_ai(
    chat_history: list,
    cfg: dict,
    prompt: str,
    data_context: str,
    skill_instruction: str = "",
) -> str:
    client = get_client(cfg)

    base_system = build_base_system()

    skill_message = (
        f"SKILL_INSTRUCTIONS:\n{skill_instruction}"
        if skill_instruction
        else "SKILL_INSTRUCTIONS:\nNone"
    )

    data_message = f"DATA_CONTEXT_JSON:\n{data_context}"

    # Avoid resending old large data contexts
    clean_history = [
        m for m in chat_history[-6:]
        if "DATA_CONTEXT_JSON" not in str(m.get("content", ""))
    ]

    messages = [
        {"role": "system", "content": base_system},
        {"role": "system", "content": skill_message},
        {"role": "user", "content": data_message},
        *clean_history,
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=cfg["deployment"],
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content

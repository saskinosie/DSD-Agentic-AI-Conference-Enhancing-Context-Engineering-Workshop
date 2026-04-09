import openai
import os

from pydantic_ai import Agent
from qdrant_client.models import Filter, FieldCondition, MatchValue


billing_agent = Agent(
    model="openai:gpt-4.1-mini",
    instructions="""
    You are a billing and payments support agent for an e-commerce clothing store.
    You help users with questions about charges, refunds, discounts, and order payments.

    You do NOT have access to live order data — acknowledge the user's issue clearly,
    explain what steps they should take (e.g. contacting support, checking their email),
    and be empathetic and helpful.

    Keep responses concise and actionable.
    """,
)

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _embed_text(text: str) -> list[float]:
    client = _get_openai_client()
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


async def handle_billing_query(qdrant, collection_name: str, slots: dict[str, str]) -> str:
    # Billing queries don't require a vector search — respond based on the slots directly
    issue = slots.get("billing_issue", "billing question")
    order_ref = slots.get("order_reference")
    amount = slots.get("amount_keyword")

    prompt = f"The user has a billing issue: {issue}."
    if order_ref:
        prompt += f" They referenced order: {order_ref}."
    if amount:
        prompt += f" Amount mentioned: {amount}."
    prompt += " Respond helpfully and tell them what steps to take."

    result = await billing_agent.run(prompt)
    return result.output

import openai
import os

from pydantic_ai import Agent
from qdrant_client.models import Filter, FieldCondition, MatchValue


support_agent = Agent(
    model="openai:gpt-4.1-mini",
    instructions="""
    You are a customer support agent for an e-commerce clothing store.
    You help users resolve issues with their orders — damaged items, wrong items,
    delivery problems, returns, and exchanges.

    You do NOT have access to live order data — acknowledge the user's issue clearly,
    be empathetic, and guide them through the appropriate next steps
    (e.g. initiating a return, contacting the courier, taking photos of damage).

    Keep responses concise, warm, and actionable.
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


async def handle_support_query(qdrant, collection_name: str, slots: dict[str, str]) -> str:
    # Support queries don't require a vector search — respond based on the issue directly
    issue = slots.get("issue_keyword", "support issue")
    order_ref = slots.get("order_reference")
    product = slots.get("product_keyword")

    prompt = f"The user has reported an issue: {issue}."
    if product:
        prompt += f" The affected item is: {product}."
    if order_ref:
        prompt += f" They referenced order: {order_ref}."
    prompt += " Respond empathetically and guide them through the next steps to resolve it."

    result = await support_agent.run(prompt)
    return result.output

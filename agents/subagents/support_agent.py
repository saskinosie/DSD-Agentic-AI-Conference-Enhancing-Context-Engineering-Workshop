import openai
import os

from pydantic_ai import Agent
from qdrant_client.models import Filter, FieldCondition, MatchValue


support_agent = Agent(
    model="openai:gpt-4.1-mini",
    instructions="""
    You are a customer support specialist agent. You help users find products
    that match specific requirements, troubleshoot product searches, and
    provide detailed product information.

    When presenting results, focus on:
    - Detailed product descriptions
    - Material and construction details when available
    - Size and fit guidance from the descriptions
    - Relevant product alternatives

    Use clear, helpful language. If the available products don't match well,
    say so and suggest what to search for instead.
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
    filter_conditions = []

    if slots.get("author"):
        filter_conditions.append(
            FieldCondition(key="department_name", match=MatchValue(value=slots["author"]))
        )

    if slots.get("contract_type"):
        filter_conditions.append(
            FieldCondition(key="product_type_name", match=MatchValue(value=slots["contract_type"]))
        )

    query_filter = Filter(must=filter_conditions) if filter_conditions else None
    search_query = slots.get("issue_keyword", "product details specifications help")
    query_vector = _embed_text(search_query)

    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=5,
    ).points

    if not results:
        return "No products found matching your support query. Try broadening your search."

    context = "\n\n---\n\n".join(
        f"Product: {hit.payload['prod_name']} ({hit.payload['product_type_name']})\n"
        f"Color: {hit.payload['colour_group_name']} | Section: {hit.payload['section_name']}\n"
        f"Description: {hit.payload['detail_desc']}"
        for hit in results
    )

    result = await support_agent.run(
        f"Based on these products, answer the user's support question.\n\n"
        f"User's focus: {slots}\n\n"
        f"Product context:\n{context}"
    )

    return result.output

import openai
import os

from pydantic_ai import Agent
from qdrant_client.models import Filter, FieldCondition, MatchValue


product_agent = Agent(
    model="openai:gpt-4.1-mini",
    instructions="""
    You are a product discovery specialist agent. You help users find
    and compare products across categories, styles, and departments.

    When presenting results, focus on:
    - Product names and types
    - Color and style details
    - Department and section context
    - How products compare to each other

    Present information in a structured, easy-to-scan format.
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


async def handle_product_query(qdrant, collection_name: str, slots: dict[str, str]) -> str:
    filter_conditions = []

    # Only filter on colour if it looks like a single canonical value (no spaces/conjunctions).
    # The slot filler sometimes returns phrases like "red or burgundy" which won't match any
    # single DB value, so we skip those and let the vector search handle colour semantics.
    colour = slots.get("colour", "")
    if colour and " " not in colour and len(colour) < 20:
        filter_conditions.append(
            FieldCondition(key="colour_group_name", match=MatchValue(value=colour))
        )

    # category and section are intentionally excluded from hard filters — the slot filler
    # extracts natural language (e.g. "women", "light") that won't exactly match DB canonical
    # values (e.g. "Ladieswear", "Divided"). Vector search handles these semantically.

    query_filter = Filter(must=filter_conditions) if filter_conditions else None
    # Build a richer query string so the vector search uses all slot context
    parts = [slots.get("product_keyword", "clothing")]
    if slots.get("category"):
        parts.append(slots["category"])
    if slots.get("section"):
        parts.append(slots["section"])
    if slots.get("colour"):
        parts.append(slots["colour"])
    search_query = " ".join(parts)
    query_vector = _embed_text(search_query)

    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=5,
    ).points

    if not results:
        return "No products found matching your query. Try broadening your search."

    context = "\n\n---\n\n".join(
        f"Product: {hit.payload['prod_name']} ({hit.payload['product_type_name']})\n"
        f"Color: {hit.payload['colour_group_name']} | Section: {hit.payload['section_name']}\n"
        f"Description: {hit.payload['detail_desc'][:300]}"
        for hit in results
    )

    result = await product_agent.run(
        f"Based on these products, answer the user's product question.\n\n"
        f"User's focus: {slots}\n\n"
        f"Product context:\n{context}"
    )

    return result.output

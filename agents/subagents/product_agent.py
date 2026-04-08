import openai
import os

from pydantic_ai import Agent


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


async def handle_product_query(conn, slots: dict[str, str]) -> str:
    conditions = []
    params = []

    if slots.get("contract_type"):
        conditions.append("product_type_name = %s")
        params.append(slots["contract_type"])

    if slots.get("author"):
        conditions.append("department_name = %s")
        params.append(slots["author"])

    search_query = slots.get("feature_keyword", "product style features")
    query_vector = _embed_text(search_query)

    where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(str(query_vector))

    sql = f"""
        SELECT prod_name, product_type_name, colour_group_name,
               section_name, detail_desc
        FROM products
        {where_clause}
        ORDER BY embedding <=> %s
        LIMIT 5
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        results = cur.fetchall()

    if not results:
        return "No products found matching your query. Try broadening your search."

    context = "\n\n---\n\n".join(
        f"Product: {row[0]} ({row[1]})\n"
        f"Color: {row[2]} | Section: {row[3]}\n"
        f"Description: {row[4][:300]}"
        for row in results
    )

    result = await product_agent.run(
        f"Based on these products, answer the user's product question.\n\n"
        f"User's focus: {slots}\n\n"
        f"Product context:\n{context}"
    )

    return result.output

from pydantic import BaseModel
from pydantic_ai import Agent


class IntentResult(BaseModel):
    intent: str
    confidence: float
    reasoning: str


SUPPORTED_INTENTS = ["billing", "product", "support", "unknown"]

intent_classifier = Agent(
    model="openai:gpt-4.1-mini",
    output_type=IntentResult,
    instructions=f"""
    You classify user utterances into one of these intents: {SUPPORTED_INTENTS}

    Intent definitions:
    - billing: Questions about payment terms, invoices, fees, salary, compensation,
      pricing, or financial obligations in contracts.
    - product: Questions about what products or services are described in contracts,
      contract types available, features, or specifications.
    - support: Questions about contract terms, termination clauses, dispute resolution,
      legal obligations, or help understanding contract language.
    - unknown: The query does not clearly fit any of the above intents.

    Return the most likely intent, a confidence score between 0 and 1,
    and a brief reasoning for your classification.

    Be conservative with confidence — only score above 0.8 if the intent is very clear.
    If the query could reasonably be two intents, pick the stronger one but lower the confidence.
    """,
)


async def classify_intent(utterance: str) -> IntentResult:
    result = await intent_classifier.run(f"Classify this user message: {utterance}")
    return result.output

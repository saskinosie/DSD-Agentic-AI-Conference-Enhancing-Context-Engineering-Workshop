from pydantic import BaseModel
from pydantic_ai import Agent


# Each intent has a set of required and optional slots
SLOT_DEFINITIONS = {
    "product": {
        "required": ["product_keyword"],
        "optional": ["colour", "category", "section"],
        "descriptions": {
            "product_keyword": "What the user is looking for (e.g. 'jacket', 'summer dress', 'running shoes')",
            "colour": "A color preference (e.g. 'black', 'dark blue', 'neutral tones')",
            "category": "The gender/age category (e.g. 'Menswear', 'Ladieswear', 'Kids')",
            "section": "A specific section or department (e.g. 'Sport', 'Basics', 'Divided')",
        },
    },
    "billing": {
        "required": ["billing_issue"],
        "optional": ["order_reference", "amount_keyword"],
        "descriptions": {
            "billing_issue": "The nature of the billing question (e.g. 'refund', 'charge', 'discount code')",
            "order_reference": "An order number or reference the user mentions",
            "amount_keyword": "Any amount or pricing detail mentioned (e.g. '£49.99', 'overcharged')",
        },
    },
    "support": {
        "required": ["issue_keyword"],
        "optional": ["order_reference", "product_keyword"],
        "descriptions": {
            "issue_keyword": "The problem the user is reporting (e.g. 'broken zipper', 'wrong size', 'not delivered')",
            "order_reference": "An order number or reference the user mentions",
            "product_keyword": "The product the issue relates to (e.g. 'jacket', 'dress')",
        },
    },
}


class SlotExtractionResult(BaseModel):
    extracted_slots: dict[str, str]
    missing_required: list[str]
    clarifying_question: str | None


slot_extractor = Agent(
    model="openai:gpt-4.1-mini",
    output_type=SlotExtractionResult,
    instructions="""
    You extract slot values from a user message for a given intent.

    You will receive:
    1. The user's message
    2. The current intent
    3. The slot definitions (required and optional slots with descriptions)
    4. Any slots already filled from previous turns

    Your job:
    - Extract any slot values present in the user's message
    - Identify which required slots are still missing
    - If required slots are missing, generate a natural clarifying question
    - NEVER hallucinate slot values — only extract what the user actually said
    - If a slot value is ambiguous, ask for clarification rather than guessing

    Return extracted_slots as a dict of slot_name: value pairs.
    Return missing_required as a list of slot names that are required but not yet filled.
    Return clarifying_question as a natural question to ask, or null if all required slots are filled.
    """,
)


async def extract_slots(
    utterance: str,
    intent: str,
    existing_slots: dict[str, str] | None = None,
) -> SlotExtractionResult:
    if existing_slots is None:
        existing_slots = {}

    slot_def = SLOT_DEFINITIONS.get(intent, {})

    prompt = f"""
    User message: "{utterance}"
    Intent: {intent}
    Slot definitions: {slot_def}
    Already filled slots: {existing_slots}

    Extract slot values from the user message and identify missing required slots.
    """

    result = await slot_extractor.run(prompt)
    return result.output

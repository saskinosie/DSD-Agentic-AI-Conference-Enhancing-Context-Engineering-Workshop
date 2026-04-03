from pydantic import BaseModel
from pydantic_ai import Agent


# Each intent has a set of required and optional slots
SLOT_DEFINITIONS = {
    "billing": {
        "required": ["author"],
        "optional": ["date_range", "contract_type", "amount_keyword"],
        "descriptions": {
            "author": "The person or entity whose contracts to search",
            "date_range": "A time period to filter by (e.g. 'after 2023', 'last year')",
            "contract_type": "The type of contract (e.g. 'employment', 'service agreement')",
            "amount_keyword": "Keywords related to amounts (e.g. 'salary', 'fee', 'payment')",
        },
    },
    "product": {
        "required": ["contract_type"],
        "optional": ["author", "feature_keyword"],
        "descriptions": {
            "contract_type": "The type of contract to search for (e.g. 'partnership', 'service agreement')",
            "author": "The person or entity who authored the contract",
            "feature_keyword": "Specific features or terms to look for",
        },
    },
    "support": {
        "required": ["issue_keyword"],
        "optional": ["author", "contract_type"],
        "descriptions": {
            "issue_keyword": "The specific issue or clause (e.g. 'termination', 'dispute', 'confidentiality')",
            "author": "The person or entity whose contracts to search",
            "contract_type": "The type of contract to search",
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

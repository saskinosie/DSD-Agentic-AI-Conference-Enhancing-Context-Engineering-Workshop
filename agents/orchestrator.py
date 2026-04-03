from dataclasses import dataclass, field

from agents.intent_classifier import classify_intent, IntentResult
from agents.slot_filler import extract_slots, SlotExtractionResult
from agents.subagents.billing_agent import handle_billing_query
from agents.subagents.product_agent import handle_product_query
from agents.subagents.support_agent import handle_support_query


CONFIDENCE_THRESHOLD = 0.6
INTENT_SWITCH_THRESHOLD = 0.7


@dataclass
class SessionState:
    current_intent: str | None = None
    intent_confidence: float = 0.0
    filled_slots: dict[str, str] = field(default_factory=dict)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    turn_count: int = 0

    def reset_intent(self):
        """Clean state reset when intent switches."""
        self.current_intent = None
        self.intent_confidence = 0.0
        self.filled_slots = {}

    def add_turn(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        if role == "user":
            self.turn_count += 1


SUBAGENT_HANDLERS = {
    "billing": handle_billing_query,
    "product": handle_product_query,
    "support": handle_support_query,
}


async def process_turn(
    session: SessionState,
    utterance: str,
    qdrant,
    collection_name: str = "financial_contracts",
) -> str:
    """
    Process a single user turn through the full orchestration pipeline:
    1. Classify intent (or continue current intent)
    2. Extract/fill slots
    3. If all required slots filled, route to subagent
    4. If not, ask clarifying question
    """
    session.add_turn("user", utterance)

    # Step 1: Intent classification
    intent_result: IntentResult = await classify_intent(utterance)

    print(f"  [Intent] {intent_result.intent} (confidence: {intent_result.confidence:.2f})")
    print(f"  [Reasoning] {intent_result.reasoning}")

    # Decide whether to switch intents or continue current conversation
    if session.current_intent is None:
        # First turn — accept the classification
        if intent_result.confidence < CONFIDENCE_THRESHOLD:
            response = (
                f"I'm not quite sure what you're looking for. "
                f"I can help with billing questions, product/contract inquiries, "
                f"or support with contract terms. Could you tell me more about what you need?"
            )
            session.add_turn("assistant", response)
            return response
        session.current_intent = intent_result.intent
        session.intent_confidence = intent_result.confidence
    elif (
        intent_result.intent != session.current_intent
        and intent_result.confidence > INTENT_SWITCH_THRESHOLD
    ):
        # Intent switch detected — clean state reset
        print(f"  [Switch] {session.current_intent} -> {intent_result.intent}")
        session.reset_intent()
        session.current_intent = intent_result.intent
        session.intent_confidence = intent_result.confidence
    # else: continue with current intent (prevents over-eager reclassification)

    # Step 2: Slot filling
    slot_result: SlotExtractionResult = await extract_slots(
        utterance=utterance,
        intent=session.current_intent,
        existing_slots=session.filled_slots,
    )

    # Merge newly extracted slots into session
    session.filled_slots.update(slot_result.extracted_slots)

    print(f"  [Slots] Filled: {session.filled_slots}")
    print(f"  [Slots] Missing required: {slot_result.missing_required}")

    # Step 3: Check if we have enough to query
    if slot_result.missing_required:
        # Still need more info — ask the clarifying question
        response = slot_result.clarifying_question or (
            f"I need a bit more information. Could you provide: "
            f"{', '.join(slot_result.missing_required)}?"
        )
        session.add_turn("assistant", response)
        return response

    # Step 4: Route to the appropriate subagent
    handler = SUBAGENT_HANDLERS.get(session.current_intent)
    if handler is None:
        response = "I'm not sure how to handle that request. Can you rephrase?"
        session.add_turn("assistant", response)
        return response

    print(f"  [Route] -> {session.current_intent} subagent")

    response = await handler(qdrant, collection_name, session.filled_slots)
    session.add_turn("assistant", response)
    return response

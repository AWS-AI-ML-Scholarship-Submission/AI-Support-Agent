"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
try:
    from strands_tools.browser import AgentCoreBrowser
except Exception:
    AgentCoreBrowser = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()

os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ── TODO 2 — Configuration ────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-nyfwwnndfw.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "GNIL4WKOTQ"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-61zm5DFW3N"

SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce
platform. You can look up orders, process refunds, answer product/policy
questions using the knowledge base, calculate loyalty discounts, and browse
the web when needed. Be concise, accurate, and always confirm order or
customer identifiers before taking action. If you do not have enough
information to help, ask a clarifying question."""

# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"
print("[INIT] Creating BedrockModel...")
model = BedrockModel(model_id=model_id)
print("[INIT] Creating MemoryClient...")
memory_client = MemoryClient(region_name=REGION)
print("[INIT] Creating boto3 client...")
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)
print("[INIT] All clients ready")

# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type to namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    return {strategy["type"]: strategy["namespaces"][0] for strategy in strategies}


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(self, actor_id: str, session_id: str, memory_client: MemoryClient, memory_id: str):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self._namespaces = None

    @property
    def namespaces(self):
        if self._namespaces is None:
            try:
                self._namespaces = get_namespaces(self.memory_client, self.memory_id)
                print(f"[MEMORY] Namespaces loaded: {self._namespaces}")
            except Exception as e:
                logger.warning("Could not fetch namespaces: %s — using defaults", e)
                self._namespaces = {
                    "SEMANTIC": "cs_agent/{actorId}/facts",
                    "USER_PREFERENCE": "cs_agent/{actorId}/preferences",
                }
        return self._namespaces

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return
        content = last_message.get("content") or []
        if not content:
            return
        first_block = content[0]
        if "toolResult" in first_block:
            return
        user_query = first_block.get("text", "")
        if not user_query:
            return
        memory_lines = []
        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_query,
                    top_k=5,
                )
            except Exception as e:
                logger.warning("Memory retrieval failed for %s: %s", namespace, e)
                continue
            for m in memories:
                text = (m.get("content") or {}).get("text", "")
                if text:
                    memory_lines.append(f"[{strategy_type}] {text}")
        if memory_lines:
            context_block = "\n".join(memory_lines)
            first_block["text"] = f"Customer Context:\n{context_block}\n\n{user_query}"

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages
        customer_query = None
        agent_response = None
        for msg in reversed(messages):
            role = msg.get("role")
            content = msg.get("content") or []
            if role == "assistant" and agent_response is None:
                for block in content:
                    if "text" in block:
                        agent_response = block["text"]
                        break
            if role == "user" and customer_query is None:
                if content and "text" in content[0]:
                    customer_query = content[0]["text"]
            if customer_query and agent_response:
                break
        if not (customer_query and agent_response):
            return
        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")],
            )
        except Exception as e:
            logger.warning("Failed to save support interaction: %s", e)

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."
    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as e:
        logger.warning("Knowledge base retrieve failed: %s", e)
        return f"Knowledge base search failed: {e}"
    results = resp.get("retrievalResults", [])
    if not results:
        return "No relevant information found in the knowledge base."
    chunks = [
        r["content"]["text"]
        for r in results
        if r.get("content", {}).get("text")
    ]
    if not chunks:
        return "No relevant information found in the knowledge base."
    return "\n---\n".join(chunks)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json
earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}
loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"
points_available = (loyalty_points // 500) * 500
max_redeemable_value = order_total * 0.5
points_value = points_available / 100
if points_value > max_redeemable_value:
    points_value = (int(max_redeemable_value * 100) // 500) * 500 / 100
    points_redeemed = int(points_value * 100)
else:
    points_redeemed = points_available
subtotal_after_points = round(order_total - points_value, 2)
tier_rate = tier_rates.get(tier, 0.0)
tier_discount_amount = round(subtotal_after_points * tier_rate, 2)
final_total = round(subtotal_after_points - tier_discount_amount, 2)
total_savings = round(order_total - final_total, 2)
earn_rate = earn_rates.get(product_category, 1)
points_earned = int(final_total * earn_rate)
remaining_points = loyalty_points - points_redeemed + points_earned
result = {{
    "order_total": order_total,
    "tier": tier,
    "tier_discount_rate": tier_rate,
    "tier_discount_amount": tier_discount_amount,
    "points_redeemed": points_redeemed,
    "points_value": points_value,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}
print(json.dumps(result))
"""
    try:
        with code_session(REGION) as session:
            response = session.invoke(
                "executeCode",
                {"code": code, "language": "python", "clearContext": True},
            )
            for event in response["stream"]:
                result_event = event.get("result")
                if result_event:
                    return json.dumps(result_event, default=str)
            return json.dumps({"error": "No result returned from code interpreter."})
    except Exception as e:
        logger.warning("Code interpreter unavailable, using fallback: %s", e)
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_rate = tier_rates.get(tier, 0.0)
        tier_discount_amount = round(order_total * tier_rate, 2)
        final_total = round(order_total - tier_discount_amount, 2)
        return json.dumps({
            "warning": "Code interpreter unavailable — tier-only fallback used.",
            "order_total": order_total,
            "tier": tier,
            "tier_discount_rate": tier_rate,
            "tier_discount_amount": tier_discount_amount,
            "final_total": final_total,
        })


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous")
        session_id = payload.get("session_id") or str(uuid.uuid4())
        print(f"[1/6] Starting agent for customer {actor_id}, session {session_id}")

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )
        print("[2/6] Memory hook initialized")

        tools = [search_knowledge_base, calculate_loyalty_discount]
        if AgentCoreBrowser is not None:
            agent_core_browser = AgentCoreBrowser(region=REGION)
            tools.append(agent_core_browser.browser)
        print(f"[3/6] Base tools loaded: {len(tools)}")

        mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        with mcp_client:
            print("[4/6] Connected to Gateway, loading tools...")
            gateway_tools = mcp_client.list_tools_sync()
            tools.extend(gateway_tools)
            print(f"[5/6] Gateway tools loaded: {len(gateway_tools)} tools")

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT,
            )

            print("[6/6] Invoking agent...")
            result = await agent.invoke_async(user_input)

        return result.message["content"][0]["text"]
   
    except Exception as e:
        logger.exception("Agent invocation failed")
        return f"I'm sorry, something went wrong while processing your request: {e}"

# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
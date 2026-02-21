---
name: 12-Factor Agents Implementation
description: Use this skill when asked to design, architect, or build LLM-powered autonomous agents, or when the user wants to make an existing agent production-ready, reliable, and scalable (e.g., using tool calling, state machines, or LangGraph).
---
# 12-Factor Agents Principles
This skill provides the standard guidelines for implementing AI agents that are robust enough for production environments, moving away from fragile experimental prototypes.
## Core Principles
1. **Natural Language to Tool Calls (JSON Extraction is Your Superpower)**
   - Always convert unstructured natural language into strictly structured data (e.g., JSON or Pydantic models). The AI should output a verified schema rather than executing tools directly.
   
2. **Tools are just Structured Outputs**
   - Think of "tool usage" simply as the LLM returning a valid payload. The actual execution of the tool (calling the API or DB) should be handled by deterministic host code.
3. **Own Your Prompts**
   - Treat prompts as code. They must be explicitly defined and version-controlled (e.g., in git). Avoid deep abstractions that obscure the actual prompts sent to the model.
4. **Own Your Context Window**
   - Actively manage what the LLM sees. Summarize context, truncate logs, and filter irrelevant history to avoid the "lost-in-the-middle" effect and token limits.
5. **Small, Focused Agents Beat Monoliths**
   - Build specialized micro-agents with a single, clear responsibility (e.g., a "Research Agent" or "Scheduling Agent") instead of monolithic, omnipotent agents.
6. **Make your agent a Stateless Reducer**
   - Agents should not maintain hidden internal states. They should act like pure functions: take current state as input, return a new state, and persist that state externally.
7. **Launch/Pause/Resume with Simple APIs**
   - Agents should be capable of pausing their workflow when waiting for an external event (human input, slow API) and securely resuming from the saved state.
8. **Contact Humans with Tool Calls**
   - Treat human intervention and approval as first-class operations. When the agent is uncertain, it should trigger an `ask_human` tool call to get user clarification.
9. **Unify Execution State and Business State (Clear Separation)**
   - Maintain a solid architectural boundary between an agent's internal "scratchpad" reasoning (execution state) and the actual business data presented to users.
10. **Compact Errors into Context Window**
    - When self-correction is needed, do not dump massive stack traces to the LLM. Parse the errors into concise, relevant messages to feed back into the context.
11. **Trigger from Anywhere, Meet Users Where They Are**
    - Expose the agent via simple APIs or event listeners so it can be integrated into Slack, CI/CD pipelines, or any other platform independently of its conversational UI.
12. **Own Your Control Flow**
    - Do not delegate the application's entire control loop to the LLM. Use deterministic code (`if` statements, state machines, LangGraph) to ensure strict state transitions and predictability.
## Implementation Guidelines
- **Schema Definition**: Make heavy use of typed schemas (e.g., Pydantic) to strictly define the `AgentDecision` or expected output.
- **Workflow State Machines**: Use frameworks that promote node-based state machines (like LangGraph) so that routing strictly resides in python/JS code, not in the LLM's imagination.
- **Human-in-the-loop**: Build the UI/UX around the agent gracefully pausing, saving its state to a database, and reloading it when the human provides an answer.
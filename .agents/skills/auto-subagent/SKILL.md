---
name: auto-subagent
description: Automatically delegates complex, multi-step, parallelizable, research, or browser-based tasks to subagents.
---

# Auto Sub-Agent Skill

This skill guides the agent to automatically evaluate, decompose, and delegate tasks to subagents when faced with eligible workloads.

## Trigger Conditions
Automatically trigger subagent delegation when receiving requests involving:
1. **Web Browser Automation / UI Testing**: Any task requiring web browsing, web page navigation, visual inspection, or interactive web operations (use `browser_subagent`).
2. **Complex & Modular Coding Tasks**: Large features or refactoring tasks that can be broken into independent sub-components.
3. **Parallel Task Execution**: Running concurrent investigations, background builds, or multi-source documentation gathering.
4. **Isolated Research & Verification**: Independent verification or diagnostic steps that benefit from an isolated context.

## Execution Workflow

### 1. Task Evaluation & Decomposition
- Evaluate if the user prompt or active objective meets trigger conditions.
- Deconstruct the goal into modular, self-contained sub-tasks with clear input/output contracts.

### 2. Subagent Delegation
When invoking a subagent:
- Formulate a precise, comprehensive task prompt containing all required context.
- Set explicit stopping criteria and define exact deliverable requirements.
- Specify human-readable task names and summaries for UI tracking.

### 3. Output Synthesis & Integration
- Inspect logs and returned artifacts upon subagent completion.
- Validate subagent deliverables against main objective requirements.
- Consolidate results cleanly into the primary conversation response.

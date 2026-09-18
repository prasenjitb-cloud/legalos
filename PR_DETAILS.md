# PR Title

Add LangGraph workflow foundation for RAG

## Description

This PR moves the existing RAG flow into a small LangGraph workflow without changing its behavior.

The graph uses the existing query rewriter, retriever, and answer generator:

`rewrite → retrieve → generate`

## Why

The current pipeline sends every query through the same retrieval flow. Our long-term plan is to add a query-classifier node and use different retrieval paths for direct, scenario-based, and multi-issue queries.

LangGraph gives us shared state and clear routing between nodes. Adding this small foundation now will keep the pipeline understandable as query-specific retrieval and reasoning steps are introduced later. It prepares the codebase for an agentic workflow without adding unfinished agent behavior in this PR.

## Changes

- Added LangGraph as a dependency.
- Added a typed shared state and three-node workflow.
- Reused one compiled graph for interactive and batch runs.
- Preserved the existing `run_rag()` output format.
- Added workflow tests and updated the relevant READMEs.

## Testing

- All four workflow unit tests pass.
- Python 3.11 compilation passes.
- A normal legal question completed successfully using the local vector database and Ollama model.

## Screenshot

Add a screenshot of a successful normal question run here before opening the PR.

## Not Included

- Query classification or routing.
- Chunk filtering or reranking.
- Retrieval, prompt, or answer-schema changes.

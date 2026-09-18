"""LangGraph workflow for the RAG pipeline."""

from typing import Required, TypedDict

from langgraph.graph import END, START, StateGraph

from chatbot.legalos_rag import queryRewriter, runRag
from chatbot.legalos_rag.prompt.promptSchema import LegalAnswer


class RAGState(TypedDict, total=False):
    """State shared by the RAG nodes."""

    query: Required[str]
    rewritten_queries: list[str]
    retrieved_chunks: str
    result: LegalAnswer | None
    final_prompt: str | None


def build_rag_graph(db_path: str, prompt_template: str, slm):
    """Compile the rewrite -> retrieve -> generate workflow."""

    def rewrite_query(state: RAGState) -> dict:
        # Create legal search variants from the user's question.
        queries = queryRewriter.rewrite_and_expand(state["query"], slm)
        return {"rewritten_queries": queries}

    def retrieve_chunks(state: RAGState) -> dict:
        # Retrieve context using all query variants.
        chunks = runRag.getFactsMulti(
            queries=state["rewritten_queries"],
            db_path=db_path,
        )
        return {"retrieved_chunks": chunks}

    def generate_answer(state: RAGState) -> dict:
        # Generate only when retrieval found context.
        chunks = state.get("retrieved_chunks", "")
        if not chunks:
            return {
                "result": None,
                "final_prompt": None,
            }

        result, final_prompt = runRag.invoker(
            slm,
            chunks,
            state["query"],
            prompt_template,
        )
        return {
            "result": result,
            "final_prompt": final_prompt,
        }

    # Keep the first graph linear. Routing will be added later.
    builder = StateGraph(RAGState)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("retrieve", retrieve_chunks)
    builder.add_node("generate", generate_answer)
    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()

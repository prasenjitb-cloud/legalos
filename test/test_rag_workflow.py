"""Tests for the LangGraph RAG workflow."""

import unittest
from unittest import mock

import chatbot.legalos_rag.workflow
import chatbot.main


class RAGWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.slm = object()

    @mock.patch("chatbot.legalos_rag.runRag.invoker")
    @mock.patch("chatbot.legalos_rag.runRag.getFactsMulti")
    @mock.patch("chatbot.legalos_rag.queryRewriter.rewrite_and_expand")
    def test_graph_runs_nodes_in_order_and_preserves_outputs(
        self,
        rewrite,
        retrieve,
        invoke,
    ):
        rewrite.return_value = ["original", "rewritten", "variant"]
        retrieve.return_value = "formatted facts"
        expected_result = object()
        invoke.return_value = (expected_result, "final prompt")

        graph = chatbot.legalos_rag.workflow.build_rag_graph(
            db_path="/tmp/vector-db",
            prompt_template="template",
            slm=self.slm,
        )
        state = graph.invoke({"query": "original"})

        rewrite.assert_called_once_with("original", self.slm)
        retrieve.assert_called_once_with(
            queries=["original", "rewritten", "variant"],
            db_path="/tmp/vector-db",
        )
        invoke.assert_called_once_with(
            self.slm,
            mock.ANY,
            "original",
            "template",
        )
        self.assertEqual(state["retrieved_chunks"], "formatted facts")
        self.assertIs(state["result"], expected_result)
        self.assertEqual(state["final_prompt"], "final prompt")

    @mock.patch("chatbot.legalos_rag.runRag.invoker")
    @mock.patch(
        "chatbot.legalos_rag.runRag.getFactsMulti",
        return_value="",
    )
    @mock.patch(
        "chatbot.legalos_rag.queryRewriter.rewrite_and_expand",
        return_value=["question"],
    )
    def test_graph_skips_generation_when_retrieval_is_empty(
        self,
        _rewrite,
        _retrieve,
        invoke,
    ):
        graph = chatbot.legalos_rag.workflow.build_rag_graph(
            db_path="/tmp/vector-db",
            prompt_template="template",
            slm=self.slm,
        )
        state = graph.invoke({"query": "question"})

        invoke.assert_not_called()
        self.assertEqual(state["retrieved_chunks"], "")
        self.assertIsNone(state["result"])
        self.assertIsNone(state["final_prompt"])

    def test_run_rag_reuses_a_precompiled_graph(self):
        graph = mock.Mock()
        expected_result = object()
        graph.invoke.return_value = {
            "result": expected_result,
            "retrieved_chunks": "facts",
            "final_prompt": "prompt",
            "rewritten_queries": ["question", "rewritten"],
        }

        output = chatbot.main.run_rag(
            "question",
            "/tmp/vector-db",
            "template",
            self.slm,
            rag_graph=graph,
        )

        graph.invoke.assert_called_once_with({"query": "question"})
        self.assertEqual(
            output,
            (expected_result, "facts", "prompt", ["question", "rewritten"]),
        )

    def test_run_rag_preserves_empty_retrieval_output(self):
        graph = mock.Mock()
        graph.invoke.return_value = {
            "result": None,
            "retrieved_chunks": "",
            "final_prompt": None,
            "rewritten_queries": ["question"],
        }

        output = chatbot.main.run_rag(
            "question",
            "/tmp/vector-db",
            "template",
            self.slm,
            rag_graph=graph,
        )

        self.assertEqual(output, (None, [], None, ["question"]))


if __name__ == "__main__":
    unittest.main()

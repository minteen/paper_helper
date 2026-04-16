import unittest

from core.state import create_initial_state, validate_state_keys


class TestResearchState(unittest.TestCase):
    def test_create_initial_state_populates_required_defaults(self) -> None:
        state = create_initial_state(query="Graph RAG", requirements="中文综述")

        self.assertEqual(state["query"], "Graph RAG")
        self.assertEqual(state["requirements"], "中文综述")
        self.assertEqual(state["papers_metadata"], [])
        self.assertEqual(state["search_approved"], False)
        self.assertEqual(state["outline_approved"], False)
        self.assertEqual(state["final_report"], "")

    def test_validate_state_keys_detects_missing_key(self) -> None:
        state = create_initial_state(query="A", requirements="")
        state.pop("query")

        missing = validate_state_keys(state)
        self.assertIn("query", missing)


if __name__ == "__main__":
    unittest.main()

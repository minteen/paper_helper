import asyncio
import tempfile
import unittest
from pathlib import Path

from tools.arxiv_search import ArxivSearchTool


class FakeMcpClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def call_tool(self, *, server, tool_name, arguments):
        self.calls.append(
            {"server": server, "tool_name": tool_name, "arguments": arguments}
        )
        return self.result


class TestArxivSearchTool(unittest.TestCase):
    def test_search_calls_arxiv_mcp_with_expected_arguments(self) -> None:
        fake = FakeMcpClient(result=[])
        tool = ArxivSearchTool(mcp_client=fake)

        asyncio.run(
            tool.search(
                query='"multi-agent"',
                max_results=5,
                categories=["cs.AI"],
                sort_by="relevance",
                sort_order="descending",
            )
        )

        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertEqual(call["server"], "user-arxiv-mcp-server")
        self.assertEqual(call["tool_name"], "search_papers")
        self.assertEqual(call["arguments"]["query"], '"multi-agent"')
        self.assertEqual(call["arguments"]["max_results"], 5)
        self.assertEqual(call["arguments"]["categories"], ["cs.AI"])
        self.assertEqual(call["arguments"]["sort_by"], "relevance")

    def test_search_normalizes_result_fields(self) -> None:
        fake = FakeMcpClient(
            result=[
                {
                    "id": "1234.5678",
                    "title": "Demo",
                    "authors": ["A", "B"],
                    "published": "2025-01-01",
                    "updated": "2025-01-02",
                    "summary": "s",
                    "comment": "",
                    "journal_ref": "",
                    "doi": "",
                    "primary_category": "cs.AI",
                    "categories": ["cs.AI"],
                    "pdf_url": "https://arxiv.org/pdf/1234.5678.pdf",
                }
            ]
        )
        tool = ArxivSearchTool(mcp_client=fake)

        papers = asyncio.run(tool.search(query='"demo"'))
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["id"], "1234.5678")
        self.assertEqual(papers[0]["title"], "Demo")
        self.assertIn("links", papers[0])
        self.assertTrue(len(papers[0]["links"]) > 0)

    def test_download_paper_saves_markdown_to_data_papers(self) -> None:
        fake = FakeMcpClient(result={"content": "# demo paper\ncontent"})
        tool = ArxivSearchTool(mcp_client=fake)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "data" / "papers"
            file_path = asyncio.run(
                tool.download_paper(paper_id="2401.12345", save_dir=save_dir)
            )

            self.assertTrue(file_path.exists())
            self.assertEqual(file_path.parent.name, "papers")
            self.assertIn("# demo paper", file_path.read_text(encoding="utf-8"))
            self.assertEqual(fake.calls[0]["tool_name"], "download_paper")
            self.assertEqual(fake.calls[0]["arguments"]["paper_id"], "2401.12345")


if __name__ == "__main__":
    unittest.main()

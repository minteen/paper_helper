from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol
from pathlib import Path


ARXIV_MCP_SERVER = "user-arxiv-mcp-server"
ARXIV_SEARCH_TOOL = "search_papers"
ARXIV_DOWNLOAD_TOOL = "download_paper"


class McpClientProtocol(Protocol):
    async def call_tool(
        self, *, server: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Call MCP tool and return tool output."""


class ArxivSearchTool:
    def __init__(self, mcp_client: Optional[McpClientProtocol] = None) -> None:
        if mcp_client is None:
            raise ValueError("mcp_client is required for ArxivSearchTool")
        self.mcp_client = mcp_client

    async def search(
        self,
        query: str,
        max_results: int = 20,
        categories: Optional[List[str]] = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> List[Dict[str, Any]]:
        arguments: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "sort_by": self._normalize_sort_by(sort_by, sort_order),
        }
        if categories:
            arguments["categories"] = categories

        result = await self.mcp_client.call_tool(
            server=ARXIV_MCP_SERVER,
            tool_name=ARXIV_SEARCH_TOOL,
            arguments=arguments,
        )
        return self._normalize_papers(result)

    async def download_paper(
        self, paper_id: str, save_dir: Path | str = "./data/papers"
    ) -> Path:
        result = await self.mcp_client.call_tool(
            server=ARXIV_MCP_SERVER,
            tool_name=ARXIV_DOWNLOAD_TOOL,
            arguments={"paper_id": paper_id},
        )

        content = self._extract_download_content(result)
        target_dir = Path(save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{paper_id}.md"
        file_path.write_text(content, encoding="utf-8")
        return file_path

    @staticmethod
    def _normalize_sort_by(sort_by: str, sort_order: str) -> str:
        # MCP schema currently supports only relevance/date.
        if sort_by in {"relevance", "date"}:
            return sort_by
        if sort_by in {"lastUpdatedDate", "submittedDate"}:
            return "date"
        if sort_order.lower() not in {"ascending", "descending"}:
            raise ValueError("sort_order must be 'ascending' or 'descending'")
        return "relevance"

    @staticmethod
    def _normalize_papers(result: Any) -> List[Dict[str, Any]]:
        if not result:
            return []

        papers: List[Dict[str, Any]] = []
        for item in result:
            links = []
            if item.get("pdf_url"):
                links.append({"type": "pdf", "url": item["pdf_url"]})
            if item.get("abs_url"):
                links.append({"type": "abs", "url": item["abs_url"]})

            papers.append(
                {
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "authors": item.get("authors", []),
                    "published": item.get("published", ""),
                    "updated": item.get("updated", ""),
                    "summary": item.get("summary", ""),
                    "comment": item.get("comment", ""),
                    "journal_ref": item.get("journal_ref", ""),
                    "doi": item.get("doi", ""),
                    "primary_category": item.get("primary_category", ""),
                    "categories": item.get("categories", []),
                    "links": links,
                }
            )
        return papers

    @staticmethod
    def _extract_download_content(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if isinstance(result.get("content"), str):
                return result["content"]
            if isinstance(result.get("text"), str):
                return result["text"]
        raise ValueError("Unexpected download_paper result format")

"""Document-aware REPL extension for LightOnOCR pipeline.

This module extends the base REPLEnvironment with document-specific
functions for accessing sections, figures, and structured content
from preprocessed documents.

Usage:
    from src.repl_document import DocumentREPLEnvironment

    # Create with preprocessed document
    repl = DocumentREPLEnvironment.from_document_result(
        document_result,
        config=REPLConfig(),
    )

    # Use document functions
    output, _ = repl.execute("print(sections())")
    output, _ = repl.execute("print(section(1))")
    output, _ = repl.execute("print(figures())")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.document import DocumentPreprocessResult, FigureRef, Section
from src.repl_environment import REPLEnvironment, REPLConfig


@dataclass
class DocumentContext:
    """Document context for REPL access."""

    sections: list[Section]
    figures: list[FigureRef]
    total_pages: int
    failed_pages: list[int]
    original_path: str

    @classmethod
    def from_document_result(cls, result: DocumentPreprocessResult) -> DocumentContext:
        """Create from a DocumentPreprocessResult."""
        return cls(
            sections=result.sections,
            figures=result.figures,
            total_pages=result.total_pages,
            failed_pages=result.failed_pages,
            original_path=result.original_path,
        )


class DocumentREPLEnvironment(REPLEnvironment):
    """REPL environment extended with document access functions.

    Provides these additional functions:
    - section(n): Get text from section n (1-indexed)
    - sections(): List all section titles
    - figures(section=None): List figures, optionally filtered by section
    - figure_image(figure_id): Get base64 image of a figure
    - page_range(n): Get page range for section n
    - search_sections(query): Search sections by content
    """

    def __init__(
        self,
        context: str,
        document_context: DocumentContext,
        artifacts: dict[str, Any] | None = None,
        config: REPLConfig | None = None,
        llm_primitives: Any | None = None,
        tool_registry: Any | None = None,
        script_registry: Any | None = None,
        role: str | None = None,
        **kwargs,
    ):
        """Initialize the document REPL environment.

        Args:
            context: The full text context (from document_result.to_searchable_text()).
            document_context: Structured document context with sections and figures.
            artifacts: Optional pre-existing artifacts.
            config: Optional REPL configuration.
            llm_primitives: Optional LLMPrimitives for llm_call/llm_batch.
            tool_registry: Optional ToolRegistry for TOOL() invocations.
            script_registry: Optional ScriptRegistry for SCRIPT() invocations.
            role: Role name for permission checking.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(
            context=context,
            artifacts=artifacts,
            config=config,
            llm_primitives=llm_primitives,
            tool_registry=tool_registry,
            script_registry=script_registry,
            role=role,
            **kwargs,
        )
        self.document_context = document_context

        # Add document functions to globals
        self._add_document_functions()

    def _add_document_functions(self) -> None:
        """Add document-specific functions to the REPL globals."""
        self._globals["section"] = self._section
        self._globals["sections"] = self._sections
        self._globals["figures"] = self._figures
        self._globals["figure_image"] = self._figure_image
        self._globals["page_range"] = self._page_range
        self._globals["search_sections"] = self._search_sections
        self._globals["document_info"] = self._document_info

    def _section(self, n: int) -> str:
        """Get text from specific section (1-indexed).

        Args:
            n: Section number (1-indexed).

        Returns:
            Section text with title, or error message if not found.
        """
        self._exploration_calls += 1

        if n < 1 or n > len(self.document_context.sections):
            return f"Section {n} not found. Available: 1-{len(self.document_context.sections)}"

        s = self.document_context.sections[n - 1]
        result = f"{'#' * s.level} {s.title}\n\n{s.content}"

        self._exploration_log.add_event("section", {"n": n}, result)
        return result

    def _sections(self) -> list[str]:
        """List all section titles.

        Returns:
            List of formatted section titles with numbers.
        """
        self._exploration_calls += 1

        result = []
        for i, s in enumerate(self.document_context.sections):
            prefix = "  " * (s.level - 1)
            result.append(f"{i + 1}. {prefix}{s.title} (pp. {s.page_start}-{s.page_end})")

        self._exploration_log.add_event("sections", {}, result)
        return result

    def _figures(self, section: int | None = None) -> list[dict[str, Any]]:
        """List figures, optionally filtered by section.

        Args:
            section: Optional section number to filter by (1-indexed).

        Returns:
            List of figure info dicts.
        """
        self._exploration_calls += 1

        figures = self.document_context.figures

        if section is not None:
            if section < 1 or section > len(self.document_context.sections):
                return []
            section_id = self.document_context.sections[section - 1].id
            figures = [f for f in figures if f.section_id == section_id]

        result = [
            {
                "id": f.id,
                "page": f.page,
                "section": f.section_id,
                "description": f.description or "(no description)",
            }
            for f in figures
        ]

        self._exploration_log.add_event("figures", {"section": section}, result)
        return result

    def _figure_image(self, figure_id: str) -> str | None:
        """Get base64 image of a specific figure.

        Args:
            figure_id: Figure identifier (e.g., "p1_fig0").

        Returns:
            Base64 encoded image string, or None if not found.
        """
        self._exploration_calls += 1

        for f in self.document_context.figures:
            if f.id == figure_id:
                self._exploration_log.add_event(
                    "figure_image",
                    {"figure_id": figure_id},
                    f.image_base64 or "(no image)",
                )
                return f.image_base64

        self._exploration_log.add_event("figure_image", {"figure_id": figure_id}, None)
        return None

    def _page_range(self, n: int) -> tuple[int, int] | None:
        """Get page range for section n.

        Args:
            n: Section number (1-indexed).

        Returns:
            Tuple of (start_page, end_page), or None if section not found.
        """
        if n < 1 or n > len(self.document_context.sections):
            return None

        s = self.document_context.sections[n - 1]
        return s.page_range

    def _search_sections(self, query: str) -> list[dict[str, Any]]:
        """Search sections by content.

        Args:
            query: Search query string.

        Returns:
            List of matching section info dicts.
        """
        self._exploration_calls += 1

        query_lower = query.lower()
        results = []

        for i, s in enumerate(self.document_context.sections):
            if query_lower in s.title.lower() or query_lower in s.content.lower():
                # Count matches
                title_matches = s.title.lower().count(query_lower)
                content_matches = s.content.lower().count(query_lower)

                results.append(
                    {
                        "section": i + 1,
                        "title": s.title,
                        "matches": title_matches + content_matches,
                        "pages": f"{s.page_start}-{s.page_end}",
                    }
                )

        # Sort by match count
        results.sort(key=lambda x: x["matches"], reverse=True)

        self._exploration_log.add_event("search_sections", {"query": query}, results)
        return results

    def _document_info(self) -> dict[str, Any]:
        """Get document metadata.

        Returns:
            Dictionary with document info.
        """
        return {
            "path": self.document_context.original_path,
            "total_pages": self.document_context.total_pages,
            "failed_pages": self.document_context.failed_pages,
            "sections": len(self.document_context.sections),
            "figures": len(self.document_context.figures),
        }

    def get_state(self) -> str:
        """Get a summary of current REPL state for the Root LM.

        Returns:
            String describing available variables and artifacts.
        """
        state_lines = [
            f"context: str ({len(self.context)} chars)",
            f"artifacts: {list(self.artifacts.keys()) if self.artifacts else '{}'}",
            "",
            "Document Functions:",
            f"  sections(): {len(self.document_context.sections)} sections available",
            f"  section(n): Get section n content (1-{len(self.document_context.sections)})",
            f"  figures(): {len(self.document_context.figures)} figures available",
            "  figure_image(id): Get figure base64 image",
            "  search_sections(query): Search section content",
            "  document_info(): Get document metadata",
        ]

        # Show artifact previews
        for key, value in self.artifacts.items():
            preview = str(value)[:100]
            if len(str(value)) > 100:
                preview += "..."
            state_lines.append(f"  artifacts['{key}']: {preview}")

        return "\n".join(state_lines)

    @classmethod
    def from_document_result(
        cls,
        document_result: DocumentPreprocessResult,
        config: REPLConfig | None = None,
        **kwargs,
    ) -> DocumentREPLEnvironment:
        """Create a DocumentREPLEnvironment from a preprocessing result.

        Args:
            document_result: The document preprocessing result.
            config: Optional REPL configuration.
            **kwargs: Additional arguments for REPLEnvironment.

        Returns:
            DocumentREPLEnvironment with document context.
        """
        return cls(
            context=document_result.to_searchable_text(),
            document_context=DocumentContext.from_document_result(document_result),
            config=config,
            **kwargs,
        )


def create_document_repl(
    document_result: DocumentPreprocessResult,
    config: REPLConfig | None = None,
    **kwargs,
) -> DocumentREPLEnvironment:
    """Create a document REPL environment from preprocessing result.

    This is a convenience factory function.

    Args:
        document_result: The document preprocessing result.
        config: Optional REPL configuration.
        **kwargs: Additional arguments for REPLEnvironment.

    Returns:
        DocumentREPLEnvironment with document context.
    """
    return DocumentREPLEnvironment.from_document_result(
        document_result,
        config=config,
        **kwargs,
    )

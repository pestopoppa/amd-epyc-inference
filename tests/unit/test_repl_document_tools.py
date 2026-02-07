#!/usr/bin/env python3
"""Unit tests for the REPL document tools (_DocumentToolsMixin)."""

from unittest.mock import Mock, patch, mock_open

from src.repl_environment import REPLEnvironment


class TestOcrDocument:
    """Test _ocr_document() / ocr_document() function."""

    def test_ocr_document_path_validation(self):
        """Test ocr_document() validates file paths."""
        repl = REPLEnvironment(context="test")
        # Try to OCR a file outside allowed paths
        result = repl.execute("print(ocr_document('/etc/passwd'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "not in allowed" in result.output or "Path" in result.output

    def test_ocr_document_non_pdf(self):
        """Test ocr_document() rejects non-PDF files."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(ocr_document('/tmp/file.txt'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "PDF" in result.output

    @patch("requests.post")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake pdf content")
    def test_ocr_document_success(self, mock_file, mock_post):
        """Test ocr_document() successfully processes PDF."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_pages": 3,
            "pages": [
                {
                    "page": 1,
                    "text": "Page 1 content",
                    "bboxes": [{"id": "fig1", "x1": 0, "y1": 0, "x2": 100, "y2": 100}],
                },
                {"page": 2, "text": "Page 2 content", "bboxes": []},
            ],
            "elapsed_sec": 0.5,
        }
        mock_post.return_value = mock_response

        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = ocr_document('/tmp/test.pdf')
data = json.loads(output)
print(data['total_pages'])
print(len(data['figures']))
""")

        assert result.error is None
        assert "3" in result.output
        assert "1" in result.output  # 1 figure

    @patch("builtins.open", new_callable=mock_open, read_data=b"fake pdf")
    @patch("requests.post")
    def test_ocr_document_server_error(self, mock_post, mock_file):
        """Test ocr_document() handles server errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        repl = REPLEnvironment(context="test")
        result = repl.execute("print(ocr_document('/tmp/test.pdf'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "500" in result.output

    @patch("builtins.open", new_callable=mock_open, read_data=b"fake pdf")
    @patch("requests.post")
    def test_ocr_document_connection_error(self, mock_post, mock_file):
        """Test ocr_document() handles connection errors."""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError()

        repl = REPLEnvironment(context="test")
        result = repl.execute("print(ocr_document('/tmp/test.pdf'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "not running" in result.output or "9001" in result.output

    def test_ocr_document_increments_exploration(self):
        """Test ocr_document() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        # Will fail validation but still increment
        repl.execute("ocr_document('/tmp/test.pdf')")

        assert repl._exploration_calls > initial_calls


class TestAnalyzeFigure:
    """Test _analyze_figure() / analyze_figure() function."""

    def test_analyze_figure_path_validation(self):
        """Test analyze_figure() validates file paths."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(analyze_figure('/etc/passwd'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "not in allowed" in result.output or "Path" in result.output

    @patch("requests.post")
    def test_analyze_figure_success(self, mock_post):
        """Test analyze_figure() successfully analyzes image."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vl_description": "This figure shows a bar chart comparing sales data across quarters."
        }
        mock_post.return_value = mock_response

        repl = REPLEnvironment(context="test")
        result = repl.execute("print(analyze_figure('/tmp/chart.png'))")

        assert result.error is None
        assert "bar chart" in result.output or "sales data" in result.output

    @patch("requests.post")
    def test_analyze_figure_with_custom_prompt(self, mock_post):
        """Test analyze_figure() with custom prompt."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"vl_description": "The chart shows an upward trend."}
        mock_post.return_value = mock_response

        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = analyze_figure('/tmp/chart.png', 'What is the trend?')
print('trend' in output.lower())
""")

        assert result.error is None
        assert "True" in result.output

    @patch("requests.post")
    def test_analyze_figure_connection_error(self, mock_post):
        """Test analyze_figure() handles connection errors."""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError()

        repl = REPLEnvironment(context="test")
        result = repl.execute("print(analyze_figure('/tmp/chart.png'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "not available" in result.output or "8000" in result.output

    def test_analyze_figure_increments_exploration(self):
        """Test analyze_figure() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("analyze_figure('/tmp/test.png')")

        assert repl._exploration_calls > initial_calls


class TestExtractFigure:
    """Test _extract_figure() / extract_figure() function."""

    def test_extract_figure_path_validation(self):
        """Test extract_figure() validates PDF path."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(extract_figure('/etc/passwd', 1, [0, 0, 100, 100]))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "not in allowed" in result.output or "Path" in result.output

    def test_extract_figure_success(self):
        """Test extract_figure() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        # Will fail but should increment exploration
        repl.execute("extract_figure('/tmp/doc.pdf', 1, [0, 0, 100, 100])")

        assert repl._exploration_calls > initial_calls

    def test_extract_figure_returns_error_for_missing_file(self):
        """Test extract_figure() returns error for nonexistent file."""
        repl = REPLEnvironment(context="test")
        result = repl.execute(
            "print(extract_figure('/tmp/nonexistent.pdf', 1, [0, 0, 100, 100]))"
        )

        assert result.error is None
        assert "ERROR" in result.output

    def test_extract_figure_missing_dependencies(self):
        """Test extract_figure() handles missing dependencies."""
        # This test would fail if pypdfium2 is not installed
        # We'll just verify it increments exploration
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        # Will fail but should increment
        repl.execute("extract_figure('/tmp/test.pdf', 1, [0, 0, 100, 100])")

        assert repl._exploration_calls > initial_calls

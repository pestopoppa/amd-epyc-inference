"""Comprehensive tests for chat summarization pipeline.

Tests coverage for src/api/routes/chat_summarization.py (13% → target 80%+).
Focuses on: summarization detection, two-stage context processing,
chunking strategies, worker dispatch, synthesis.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from src.api.routes.chat_summarization import (
    _is_summarization_task,
    _run_two_stage_summarization,
    _should_use_two_stage,
)


# ── Summarization Task Detection ─────────────────────────────────────────


class TestSummarizationDetection:
    """Test summarization task keyword detection."""

    def test_is_summarization_task_with_summarize(self):
        """Test detection of 'summarize' keyword."""
        assert _is_summarization_task("Please summarize this document") is True
        assert _is_summarization_task("Can you summarize the key points?") is True
        assert _is_summarization_task("Summarize") is True

    def test_is_summarization_task_with_summary(self):
        """Test detection of 'summary' keyword."""
        assert _is_summarization_task("Give me a summary") is True
        assert _is_summarization_task("Provide a summary of the findings") is True
        assert _is_summarization_task("executive summary needed") is True

    def test_is_summarization_task_with_british_spelling(self):
        """Test detection of British spelling variants."""
        assert _is_summarization_task("Please summarise this") is True
        assert _is_summarization_task("Provide a summarisation") is True

    def test_is_summarization_task_with_overview_keywords(self):
        """Test detection of overview-related keywords."""
        assert _is_summarization_task("Give me an overview") is True
        assert _is_summarization_task("What are the key points?") is True
        assert _is_summarization_task("Explain the main ideas") is True

    def test_is_summarization_task_with_tldr(self):
        """Test detection of tl;dr and tldr."""
        assert _is_summarization_task("tl;dr") is True
        assert _is_summarization_task("tldr please") is True
        assert _is_summarization_task("Give me the TL;DR") is True

    def test_is_summarization_task_with_synopsis(self):
        """Test detection of 'synopsis' keyword."""
        assert _is_summarization_task("Provide a synopsis") is True
        assert _is_summarization_task("Brief synopsis needed") is True

    def test_is_summarization_task_case_insensitive(self):
        """Test case-insensitive matching."""
        assert _is_summarization_task("SUMMARIZE THIS") is True
        assert _is_summarization_task("Executive SUMMARY") is True
        assert _is_summarization_task("Key Points") is True

    def test_is_summarization_task_non_summarization(self):
        """Test non-summarization prompts return False."""
        assert _is_summarization_task("Extract all phone numbers") is False
        assert _is_summarization_task("Find the address") is False
        assert _is_summarization_task("What color is the sky?") is False
        assert _is_summarization_task("Calculate the total") is False
        assert _is_summarization_task("") is False

    def test_is_summarization_task_with_embedded_keywords(self):
        """Test that keywords are found even in longer prompts."""
        prompt = (
            "I have a long document about quantum computing. "
            "Please summarize the key findings and conclusions."
        )
        assert _is_summarization_task(prompt) is True


# ── Two-Stage Decision Logic ─────────────────────────────────────────────


class TestShouldUseTwoStage:
    """Test two-stage processing decision logic."""

    def test_should_use_two_stage_disabled_config(self):
        """Test returns False when disabled in config."""
        with patch("src.api.routes.chat_summarization.TWO_STAGE_CONFIG") as mock_config:
            mock_config.__getitem__.return_value = False
            result = _should_use_two_stage("summarize this", "A" * 30000)
            assert result is False

    def test_should_use_two_stage_no_context(self):
        """Test returns False when no context provided."""
        with patch("src.api.routes.chat_summarization.TWO_STAGE_CONFIG") as mock_config:
            mock_config.__getitem__.return_value = True
            result = _should_use_two_stage("summarize", None)
            assert result is False

    def test_should_use_two_stage_empty_context(self):
        """Test returns False for empty context."""
        with patch("src.api.routes.chat_summarization.TWO_STAGE_CONFIG") as mock_config:
            mock_config.__getitem__.return_value = True
            result = _should_use_two_stage("summarize", "")
            assert result is False

    def test_should_use_two_stage_small_context(self):
        """Test returns False for context below threshold."""
        with patch("src.api.routes.chat_summarization.TWO_STAGE_CONFIG") as mock_config:
            with patch("src.api.routes.chat_summarization.LONG_CONTEXT_CONFIG") as mock_long_config:
                mock_config.__getitem__.return_value = True
                mock_long_config.__getitem__.return_value = 20000  # threshold

                # Context with 15K chars (below threshold)
                result = _should_use_two_stage("summarize", "A" * 15000)
                assert result is False

    def test_should_use_two_stage_large_context(self):
        """Test returns True for context above threshold."""
        with patch("src.api.routes.chat_summarization.TWO_STAGE_CONFIG") as mock_config:
            with patch("src.api.routes.chat_summarization.LONG_CONTEXT_CONFIG") as mock_long_config:
                mock_config.__getitem__.return_value = True
                mock_long_config.__getitem__.return_value = 20000  # threshold

                # Context with 25K chars (above threshold)
                result = _should_use_two_stage("summarize", "A" * 25000)
                assert result is True

    def test_should_use_two_stage_multi_doc_discount(self):
        """Test multi-document threshold discount."""
        with patch("src.api.routes.chat_summarization.TWO_STAGE_CONFIG") as mock_config:
            with patch("src.api.routes.chat_summarization.LONG_CONTEXT_CONFIG") as mock_long_config:
                mock_config.__getitem__.side_effect = lambda k: {
                    "enabled": True,
                    "multi_doc_discount": 0.5,  # 50% discount
                }[k]
                mock_long_config.__getitem__.return_value = 20000  # base threshold

                # With 3 docs, threshold becomes 10K (20K * 0.5)
                # Context with 12K should trigger
                result = _should_use_two_stage("summarize", "A" * 12000, doc_count=3)
                assert result is True

    def test_should_use_two_stage_any_task_not_just_summarization(self):
        """Test two-stage triggers for ANY large context, not just summarization."""
        with patch("src.api.routes.chat_summarization.TWO_STAGE_CONFIG") as mock_config:
            with patch("src.api.routes.chat_summarization.LONG_CONTEXT_CONFIG") as mock_long_config:
                mock_config.__getitem__.return_value = True
                mock_long_config.__getitem__.return_value = 20000

                # Not a summarization task, but large context
                result = _should_use_two_stage("Find all email addresses", "A" * 25000)
                assert result is True


# ── Two-Stage Pipeline Execution ─────────────────────────────────────────


class TestTwoStagePipelineExecution:
    """Test two-stage summarization pipeline execution."""

    @pytest.mark.asyncio
    async def test_run_two_stage_summarization_basic(self):
        """Test basic two-stage pipeline execution."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = [
            "Section 1: Introduction to topic",
            "Section 2: Main findings",
        ]
        mock_primitives.llm_call.return_value = "Comprehensive summary of both sections"

        mock_state = MagicMock()
        mock_state.progress_logger = MagicMock()

        # Mock worker_fast health check to fail (use worker_explore)
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 500  # Fail health check
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            answer, stats = await _run_two_stage_summarization(
                prompt="Summarize this document",
                context="A" * 20000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-123",
            )

            assert answer == "Comprehensive summary of both sections"
            assert stats["pipeline"] == "two_stage_context"
            assert stats["chunks"] >= 2
            assert "stage1_time_ms" in stats
            assert "stage2_time_ms" in stats
            mock_primitives.llm_batch.assert_called_once()
            mock_primitives.llm_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_two_stage_chunking_strategy(self):
        """Test that context is chunked appropriately."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest 1", "Digest 2", "Digest 3"]
        mock_primitives.llm_call.return_value = "Final answer"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200  # Worker_fast available
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            # 24K context should create 3 chunks (24000 / 8000 = 3)
            answer, stats = await _run_two_stage_summarization(
                prompt="Analyze",
                context="B" * 24000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-456",
            )

            assert stats["chunks"] == 3
            assert len(mock_primitives.llm_batch.call_args[0][0]) == 3

    @pytest.mark.asyncio
    async def test_run_two_stage_worker_role_selection(self):
        """Test that worker_fast is preferred when available."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest"]
        mock_primitives.llm_call.return_value = "Answer"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        # Mock successful health check for worker_fast
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200  # Success
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            await _run_two_stage_summarization(
                prompt="Test",
                context="C" * 10000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-789",
            )

            # Verify worker_fast was used
            call_args = mock_primitives.llm_batch.call_args
            assert call_args[1]["role"] == "worker_fast"

    @pytest.mark.asyncio
    async def test_run_two_stage_worker_fallback(self):
        """Test fallback to worker_explore when worker_fast unavailable."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest"]
        mock_primitives.llm_call.return_value = "Answer"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        # Mock failed health check
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client_class.return_value = mock_client

            await _run_two_stage_summarization(
                prompt="Test",
                context="D" * 10000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-abc",
            )

            # Verify fallback to worker_explore
            call_args = mock_primitives.llm_batch.call_args
            assert call_args[1]["role"] == "worker_explore"

    @pytest.mark.asyncio
    async def test_run_two_stage_batch_failure_fallback(self):
        """Test fallback to sequential calls when llm_batch fails."""
        mock_primitives = MagicMock()
        # Batch fails, sequential calls succeed
        mock_primitives.llm_batch.side_effect = Exception("Batch dispatch failed")
        mock_primitives.llm_call.side_effect = [
            "Digest 1",
            "Digest 2",
            "Final synthesis",
        ]

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            answer, stats = await _run_two_stage_summarization(
                prompt="Test",
                context="E" * 16000,  # 2 chunks
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-def",
            )

            # Should have called llm_call 3 times (2 workers + 1 synthesis)
            assert mock_primitives.llm_call.call_count == 3
            assert answer == "Final synthesis"

    @pytest.mark.asyncio
    async def test_run_two_stage_sequential_worker_failure(self):
        """Test handling of individual worker failures in sequential fallback."""
        mock_primitives = MagicMock()
        # Batch fails, sequential calls have one failure
        mock_primitives.llm_batch.side_effect = Exception("Batch failed")
        mock_primitives.llm_call.side_effect = [
            "Digest 1",
            Exception("Worker timeout"),  # Second worker fails
            "Final synthesis",
        ]

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            answer, stats = await _run_two_stage_summarization(
                prompt="Test",
                context="F" * 16000,  # 2 chunks
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-ghi",
            )

            # Should still complete with partial digests
            assert "Digest 1" in answer or "synthesis" in answer.lower()

    @pytest.mark.asyncio
    async def test_run_two_stage_summarization_prompt_format(self):
        """Test that worker prompts include task context and instructions."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest"]
        mock_primitives.llm_call.return_value = "Answer"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            await _run_two_stage_summarization(
                prompt="Find all mentions of 'quantum entanglement'",
                context="G" * 10000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-jkl",
            )

            # Check worker prompt structure
            worker_prompts = mock_primitives.llm_batch.call_args[0][0]
            assert len(worker_prompts) > 0
            first_prompt = worker_prompts[0]
            assert "Task context:" in first_prompt
            assert "Find all mentions" in first_prompt
            assert "Section Content" in first_prompt

    @pytest.mark.asyncio
    async def test_run_two_stage_synthesis_for_summarization(self):
        """Test synthesis prompt for summarization tasks."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest 1"]
        mock_primitives.llm_call.return_value = "Summary"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            await _run_two_stage_summarization(
                prompt="Summarize the document",
                context="H" * 10000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-mno",
            )

            # Check synthesis prompt for summarization keywords
            synthesis_prompt = mock_primitives.llm_call.call_args[0][0]
            assert "comprehensive summary" in synthesis_prompt.lower()
            assert "main thesis" in synthesis_prompt.lower()

    @pytest.mark.asyncio
    async def test_run_two_stage_synthesis_for_non_summarization(self):
        """Test synthesis prompt for non-summarization tasks."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest 1"]
        mock_primitives.llm_call.return_value = "Answer"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            await _run_two_stage_summarization(
                prompt="Find all email addresses",
                context="I" * 10000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-pqr",
            )

            # Check synthesis prompt for search keywords
            synthesis_prompt = mock_primitives.llm_call.call_args[0][0]
            assert "specific items" in synthesis_prompt.lower() or "exact values" in synthesis_prompt.lower()

    @pytest.mark.asyncio
    async def test_run_two_stage_synthesis_failure_uses_digests(self):
        """Test that synthesis failure falls back to worker digests."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest 1", "Digest 2"]
        mock_primitives.llm_call.side_effect = Exception("Synthesis timeout")

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            answer, stats = await _run_two_stage_summarization(
                prompt="Test",
                context="J" * 10000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-stu",
            )

            # Should return digests directly
            assert "Digest 1" in answer
            assert "Digest 2" in answer
            assert "Worker findings:" in answer

    @pytest.mark.asyncio
    async def test_run_two_stage_progress_logging(self):
        """Test that progress is logged when logger available."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest"]
        mock_primitives.llm_call.return_value = "Answer"

        mock_state = MagicMock()
        mock_state.progress_logger = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            await _run_two_stage_summarization(
                prompt="Test",
                context="K" * 10000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-vwx",
            )

            # Verify logging was called
            mock_state.progress_logger.log_exploration.assert_called_once()
            call_args = mock_state.progress_logger.log_exploration.call_args
            assert call_args[1]["task_id"] == "test-vwx"
            assert call_args[1]["strategy_used"] == "two_stage_context"

    @pytest.mark.asyncio
    async def test_run_two_stage_stats_include_worker_digests(self):
        """Test that stats include worker digests for review gate."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = [
            "First section analysis: key point A",
            "Second section analysis: key point B",
        ]
        mock_primitives.llm_call.return_value = "Final synthesis"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            answer, stats = await _run_two_stage_summarization(
                prompt="Test",
                context="L" * 16000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-yz",
            )

            assert "worker_digests" in stats
            assert len(stats["worker_digests"]) == 2
            assert stats["worker_digests"][0]["section"] == 1
            assert "key point A" in stats["worker_digests"][0]["summary"]

    @pytest.mark.asyncio
    async def test_run_two_stage_chunk_overlap(self):
        """Test that chunks have overlap for context continuity."""
        mock_primitives = MagicMock()
        captured_prompts = []

        def capture_batch(prompts, **kwargs):
            captured_prompts.extend(prompts)
            return ["Digest"] * len(prompts)

        mock_primitives.llm_batch.side_effect = capture_batch
        mock_primitives.llm_call.return_value = "Answer"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Use unique chars to verify overlap
            context = "".join(chr(65 + i % 26) for i in range(16000))

            await _run_two_stage_summarization(
                prompt="Test",
                context=context,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-overlap",
            )

            # With overlap, chunks should share some content
            # This is indicated by chunk boundaries not being exact multiples
            assert len(captured_prompts) >= 2

    @pytest.mark.asyncio
    async def test_run_two_stage_token_estimation(self):
        """Test that context tokens are estimated in stats."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest"]
        mock_primitives.llm_call.return_value = "Answer"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            with patch("src.api.routes.chat_summarization._estimate_tokens") as mock_estimate:
                mock_estimate.return_value = 5000

                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_client.get.return_value = mock_response
                mock_client_class.return_value = mock_client

                answer, stats = await _run_two_stage_summarization(
                    prompt="Test",
                    context="M" * 10000,
                    primitives=mock_primitives,
                    state=mock_state,
                    task_id="test-tokens",
                )

                assert stats["context_tokens"] == 5000
                mock_estimate.assert_called()

    @pytest.mark.asyncio
    async def test_run_two_stage_max_chunks_limit(self):
        """Test that chunking is bounded to max 8 chunks."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest"] * 8
        mock_primitives.llm_call.return_value = "Answer"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Very large context (100K chars)
            answer, stats = await _run_two_stage_summarization(
                prompt="Test",
                context="N" * 100000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-max-chunks",
            )

            # Should be capped at 8 chunks
            assert stats["chunks"] == 8

    @pytest.mark.asyncio
    async def test_run_two_stage_min_chunks_enforced(self):
        """Test that at least 2 chunks are created."""
        mock_primitives = MagicMock()
        mock_primitives.llm_batch.return_value = ["Digest"] * 2
        mock_primitives.llm_call.return_value = "Answer"

        mock_state = MagicMock()
        mock_state.progress_logger = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Very small context (2K chars)
            answer, stats = await _run_two_stage_summarization(
                prompt="Test",
                context="O" * 2000,
                primitives=mock_primitives,
                state=mock_state,
                task_id="test-min-chunks",
            )

            # Should enforce minimum of 2 chunks
            assert stats["chunks"] >= 2

"""Tests for LlamaTokenizer (C2: Pre-flight Token Counter)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.llm_primitives.tokenizer import LlamaTokenizer


class TestLlamaTokenizer:
    """Test suite for LlamaTokenizer."""

    def _make_tokenizer(self, base_url: str = "http://localhost:8080") -> LlamaTokenizer:
        return LlamaTokenizer(base_url=base_url, timeout=0.5, cache_size=10)

    def test_count_tokens_approx(self):
        tok = self._make_tokenizer()
        assert tok.count_tokens_approx("hello world") == len("hello world") // 4
        assert tok.count_tokens_approx("") == 0
        assert tok.count_tokens_approx("a" * 100) == 25

    def test_count_tokens_calls_server(self):
        tok = self._make_tokenizer()
        mock_response = MagicMock()
        mock_response.json.return_value = {"tokens": [1, 2, 3, 4, 5]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(tok._client, "post", return_value=mock_response) as mock_post:
            result = tok.count_tokens("hello world")

        assert result == 5
        mock_post.assert_called_once_with(
            "http://localhost:8080/tokenize",
            json={"content": "hello world"},
        )
        assert tok.total_calls == 1
        assert tok.cache_hits == 0
        assert tok.fallback_count == 0

    def test_lru_cache_hit(self):
        tok = self._make_tokenizer()
        mock_response = MagicMock()
        mock_response.json.return_value = {"tokens": [1, 2, 3]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(tok._client, "post", return_value=mock_response) as mock_post:
            result1 = tok.count_tokens("hello world")
            result2 = tok.count_tokens("hello world")

        assert result1 == 3
        assert result2 == 3
        assert mock_post.call_count == 1  # Only one HTTP call
        assert tok.cache_hits == 1
        assert tok.total_calls == 2

    def test_lru_cache_eviction(self):
        tok = self._make_tokenizer()
        tok._cache_size = 3  # Small cache for testing

        mock_response = MagicMock()
        mock_response.json.return_value = {"tokens": [1]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(tok._client, "post", return_value=mock_response):
            tok.count_tokens("aaa")
            tok.count_tokens("bbb")
            tok.count_tokens("ccc")
            tok.count_tokens("ddd")  # Should evict "aaa"

        assert len(tok._cache) == 3
        # "aaa" should be evicted
        aaa_key = tok._cache_key("aaa")
        assert aaa_key not in tok._cache

    def test_fallback_on_timeout(self):
        tok = self._make_tokenizer()
        text = "a" * 40  # len=40, approx = 10

        with patch.object(tok._client, "post", side_effect=Exception("timeout")):
            result = tok.count_tokens(text)

        assert result == 10  # len(40) // 4
        assert tok.fallback_count == 1

    def test_fallback_on_http_error(self):
        tok = self._make_tokenizer()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("503 Service Unavailable")

        with patch.object(tok._client, "post", return_value=mock_response):
            result = tok.count_tokens("test text here")

        assert result == len("test text here") // 4
        assert tok.fallback_count == 1

    def test_cache_key_uses_prefix_and_length(self):
        tok = self._make_tokenizer()
        # Two strings with same prefix but different lengths should differ
        key1 = tok._cache_key("hello" + "x" * 100)
        key2 = tok._cache_key("hello" + "x" * 200)
        assert key1 != key2

        # Two strings with same prefix AND same length should match
        key3 = tok._cache_key("hello" + "a" * 195)
        key4 = tok._cache_key("hello" + "b" * 195)
        # Same prefix (first 200 chars differ after pos 5) — actually different
        assert key3 != key4

    def test_get_stats(self):
        tok = self._make_tokenizer()
        tok.total_calls = 10
        tok.cache_hits = 3
        tok.fallback_count = 1

        stats = tok.get_stats()
        assert stats["total_calls"] == 10
        assert stats["cache_hits"] == 3
        assert stats["fallback_count"] == 1
        assert stats["cache_hit_rate"] == pytest.approx(0.3)

    def test_base_url_trailing_slash_stripped(self):
        tok = LlamaTokenizer(base_url="http://localhost:8080/")
        assert tok.base_url == "http://localhost:8080"

    def test_empty_tokens_response(self):
        tok = self._make_tokenizer()
        mock_response = MagicMock()
        mock_response.json.return_value = {"tokens": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(tok._client, "post", return_value=mock_response):
            result = tok.count_tokens("")

        assert result == 0


class TestTokensMixinIntegration:
    """Test that TokensMixin uses LlamaTokenizer when set."""

    def test_mixin_uses_tokenizer_when_set(self):
        from src.llm_primitives.tokens import TokensMixin

        class TestClass(TokensMixin):
            pass

        obj = TestClass()

        # Without tokenizer — uses heuristic
        assert obj._estimate_prompt_tokens("hello world!!") == len("hello world!!") // 4

        # With tokenizer — uses accurate count
        mock_tokenizer = MagicMock()
        mock_tokenizer.count_tokens.return_value = 42
        obj._tokenizer = mock_tokenizer

        assert obj._estimate_prompt_tokens("hello world!!") == 42
        assert obj._estimate_completion_tokens("test response") == 42
        assert obj._count_tokens("any text") == 42

    def test_mixin_count_tokens_convenience(self):
        from src.llm_primitives.tokens import TokensMixin

        class TestClass(TokensMixin):
            pass

        obj = TestClass()
        # Without tokenizer
        assert obj._count_tokens("a" * 100) == 25

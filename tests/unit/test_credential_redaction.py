"""Tests for credential redaction in tool and REPL output."""

from __future__ import annotations

import pytest

from src.repl_environment.redaction import (
    redact_credentials,
    redact_if_enabled,
)


class TestRedactCredentials:
    """Tests for the redact_credentials() core function."""

    # === AWS ===

    def test_aws_access_key_redacted(self):
        text = "Found key: AKIAIOSFODNN7EXAMPLE in config"
        result = redact_credentials(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result.text
        assert "[REDACTED:aws_access_key]" in result.text
        assert result.redacted_count == 1
        assert "aws_access_key" in result.categories

    # === Anthropic ===

    def test_anthropic_key_redacted(self):
        text = "key = sk-ant-api03-abcdefghij1234567890_ABCDEF"
        result = redact_credentials(text)
        assert "sk-ant-api03" not in result.text
        assert "[REDACTED:anthropic_key]" in result.text
        assert "anthropic_key" in result.categories

    # === OpenAI ===

    def test_openai_key_redacted(self):
        text = "The key is sk-proj1234567890abcdefghijklmno here"
        result = redact_credentials(text)
        assert "sk-proj1234567890" not in result.text
        assert "[REDACTED:openai_key]" in result.text
        assert "openai_key" in result.categories

    def test_openai_key_does_not_match_anthropic(self):
        """sk-ant- prefix should match anthropic, not openai."""
        text = "key = sk-ant-api03-abcdefghij1234567890_ABCDEF"
        result = redact_credentials(text)
        assert "anthropic_key" in result.categories
        assert "openai_key" not in result.categories

    # === GitHub tokens ===

    def test_github_pat_redacted(self):
        text = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij1234"
        result = redact_credentials(text)
        assert "ghp_ABCDEFGH" not in result.text
        assert "[REDACTED:github_pat]" in result.text
        assert "github_pat" in result.categories

    def test_github_oauth_redacted(self):
        text = "oauth: gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij1234"
        result = redact_credentials(text)
        assert "[REDACTED:github_oauth]" in result.text

    def test_github_app_tokens_redacted(self):
        text = "ghs_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij1234 and ghu_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij1234"
        result = redact_credentials(text)
        assert result.redacted_count == 2
        assert "[REDACTED:github_token]" in result.text

    def test_github_fine_grained_pat_redacted(self):
        text = "pat: github_pat_11ABCDEFG0AbCdEfGhIjKl_xyzABCDEF"
        result = redact_credentials(text)
        assert "[REDACTED:github_fine_grained_pat]" in result.text

    # === SSH private keys ===

    def test_ssh_rsa_private_key_redacted(self):
        text = (
            "Here is the key:\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/yGaXm...\n"
            "more key data here...\n"
            "-----END RSA PRIVATE KEY-----\n"
            "End of key."
        )
        result = redact_credentials(text)
        assert "MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn" not in result.text
        assert "[REDACTED:ssh_private_key]" in result.text
        assert "ssh_private_key" in result.categories

    def test_ssh_openssh_private_key_redacted(self):
        text = (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n"
            "b3BlbnNzaC1rZXktdjEAAAAACmFlczI1Ni1jdHI...\n"
            "-----END OPENSSH PRIVATE KEY-----"
        )
        result = redact_credentials(text)
        assert "[REDACTED:ssh_private_key]" in result.text

    def test_ssh_ec_private_key_redacted(self):
        text = (
            "-----BEGIN EC PRIVATE KEY-----\n"
            "MHQCAQEEIGABn...\n"
            "-----END EC PRIVATE KEY-----"
        )
        result = redact_credentials(text)
        assert "[REDACTED:ssh_private_key]" in result.text

    # === Bearer tokens ===

    def test_bearer_token_redacted(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkw"
        result = redact_credentials(text)
        assert "eyJhbGciOiJIUzI1NiI" not in result.text
        assert "Bearer [REDACTED:token]" in result.text

    # === Connection strings ===

    def test_postgres_connection_string_redacted(self):
        text = "DATABASE_URL=postgresql://user:password123@host.example.com:5432/mydb"
        result = redact_credentials(text)
        assert "password123" not in result.text
        assert "[REDACTED:" in result.text

    def test_redis_connection_string_redacted(self):
        text = "REDIS_URL=redis://default:secretpass@redis.example.com:6379/0"
        result = redact_credentials(text)
        assert "secretpass" not in result.text

    def test_mongodb_connection_string_redacted(self):
        text = "mongodb+srv://admin:password@cluster0.abc123.mongodb.net/test"
        result = redact_credentials(text)
        assert "password" not in result.text
        assert "[REDACTED:connection_string]" in result.text

    # === Env secrets ===

    def test_env_api_key_redacted(self):
        text = 'API_KEY="my-super-secret-api-key-12345"'
        result = redact_credentials(text)
        assert "my-super-secret" not in result.text
        assert "[REDACTED:env_value]" in result.text

    def test_env_secret_key_redacted(self):
        text = "SECRET_KEY=django-insecure-abc123def456ghi789"
        result = redact_credentials(text)
        assert "django-insecure" not in result.text

    def test_env_password_redacted(self):
        text = "PASSWORD=hunter2-extended-version"
        result = redact_credentials(text)
        assert "hunter2" not in result.text

    # === Slack tokens ===

    def test_slack_bot_token_redacted(self):
        text = "The token is xoxb-fake00test00val"
        result = redact_credentials(text)
        assert "xoxb-fake00test00val" not in result.text

    # === Stripe keys ===

    def test_stripe_secret_key_redacted(self):
        text = "stripe_key = sk_test_FAKEFAKEFAKEFAKEFAKE00"
        result = redact_credentials(text)
        assert "sk_test_FAKEFAKE" not in result.text
        assert "[REDACTED:stripe_key]" in result.text

    def test_stripe_publishable_key_redacted(self):
        text = "pk_test_FAKEFAKEFAKEFAKEFAKE00"
        result = redact_credentials(text)
        assert "[REDACTED:stripe_key]" in result.text

    # === Google Cloud ===

    def test_gcloud_api_key_redacted(self):
        text = "Found gcloud key AIzaSyA1234567890abcdefghijklmnopqrstuv in config"
        result = redact_credentials(text)
        assert "AIzaSyA12345" not in result.text
        assert "[REDACTED:gcloud_key]" in result.text

    # === Edge cases ===

    def test_short_string_not_scanned(self):
        """Strings under 16 chars should not be scanned."""
        result = redact_credentials("short")
        assert result.text == "short"
        assert result.redacted_count == 0

    def test_non_string_input(self):
        result = redact_credentials(42)  # type: ignore[arg-type]
        assert result.text == "42"
        assert result.redacted_count == 0

    def test_empty_string(self):
        result = redact_credentials("")
        assert result.text == ""
        assert result.redacted_count == 0

    def test_normal_code_no_false_positives(self):
        """Normal Python code should not trigger redaction."""
        code = """
def calculate_sum(numbers):
    total = sum(numbers)
    print(f"Sum: {total}")
    return total

result = calculate_sum([1, 2, 3, 4, 5])
print(f"Result: {result}")
"""
        result = redact_credentials(code)
        assert result.redacted_count == 0
        assert result.text == code

    def test_normal_shell_output_no_false_positives(self):
        """Typical shell command output should not be redacted."""
        output = """
total 48K
drwxr-xr-x 5 user user 4.0K Feb 18 14:30 src/
drwxr-xr-x 3 user user 4.0K Feb 18 14:30 tests/
-rw-r--r-- 1 user user 8.5K Feb 18 14:30 README.md
-rw-r--r-- 1 user user 2.1K Feb 18 14:30 pyproject.toml
"""
        result = redact_credentials(output)
        assert result.redacted_count == 0

    def test_git_sha_not_redacted(self):
        """40-char hex git SHAs should not be redacted (too short for hex_secret)."""
        output = "commit abc123def456789012345678901234567890abcd"
        result = redact_credentials(output)
        assert result.redacted_count == 0

    def test_multiple_credentials_in_same_text(self):
        text = (
            "AWS: AKIAIOSFODNN7EXAMPLE\n"
            "GH: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij1234\n"
            "Anthropic: sk-ant-api03-abcdefghij1234567890_ABCDEF\n"
        )
        result = redact_credentials(text)
        assert result.redacted_count == 3
        assert "aws_access_key" in result.categories
        assert "github_pat" in result.categories
        assert "anthropic_key" in result.categories

    def test_idempotent_redaction(self):
        """Redacting already-redacted text should not change it further."""
        text = "key = sk-ant-api03-abcdefghij1234567890_ABCDEF"
        first = redact_credentials(text)
        second = redact_credentials(first.text)
        assert first.text == second.text
        assert second.redacted_count == 0

    def test_preserves_surrounding_text(self):
        text = "Before the key AKIAIOSFODNN7EXAMPLE and after the key"
        result = redact_credentials(text)
        assert result.text == "Before the key [REDACTED:aws_access_key] and after the key"

    def test_frozen_result_dataclass(self):
        result = redact_credentials("some text that is long enough to scan")
        with pytest.raises(AttributeError):
            result.text = "modified"  # type: ignore[misc]


class TestRedactIfEnabled:
    """Tests for the feature-flag-aware convenience function."""

    def test_redacts_when_flag_enabled(self, monkeypatch):
        """With flag on (default), credentials should be redacted."""
        from src.features import Features, set_features, reset_features

        try:
            set_features(Features(credential_redaction=True))
            text = "key = sk-ant-api03-abcdefghij1234567890_ABCDEF"
            result = redact_if_enabled(text)
            assert "[REDACTED:anthropic_key]" in result
        finally:
            reset_features()

    def test_passthrough_when_flag_disabled(self, monkeypatch):
        """With flag off, credentials should pass through unchanged."""
        from src.features import Features, set_features, reset_features

        try:
            set_features(Features(credential_redaction=False))
            text = "key = sk-ant-api03-abcdefghij1234567890_ABCDEF"
            result = redact_if_enabled(text)
            assert result == text
            assert "sk-ant-api03" in result
        finally:
            reset_features()

    def test_short_text_passthrough(self):
        """Short text should pass through without scanning."""
        assert redact_if_enabled("hello") == "hello"


class TestToolRegistryRedaction:
    """Integration test: verify ToolRegistry.invoke() redacts output."""

    def test_tool_string_result_redacted(self):
        from src.features import Features, set_features, reset_features
        from src.tool_registry import (
            Tool,
            ToolCategory,
            ToolPermissions,
            ToolRegistry,
        )

        try:
            set_features(Features(credential_redaction=True))

            registry = ToolRegistry()

            # Register a tool that returns a credential
            tool = Tool(
                name="leaky_tool",
                description="Returns a secret",
                category=ToolCategory.DATA,
                parameters={},
                handler=lambda: "Found: AKIAIOSFODNN7EXAMPLE in env",
            )
            registry.register_tool(tool)
            registry.set_role_permissions(
                "test_role",
                ToolPermissions(allowed_categories=[ToolCategory.DATA]),
            )

            result = registry.invoke("leaky_tool", role="test_role")
            assert "AKIAIOSFODNN7EXAMPLE" not in result
            assert "[REDACTED:aws_access_key]" in result
        finally:
            reset_features()

    def test_tool_non_string_result_unchanged(self):
        from src.features import Features, set_features, reset_features
        from src.tool_registry import (
            Tool,
            ToolCategory,
            ToolPermissions,
            ToolRegistry,
        )

        try:
            set_features(Features(credential_redaction=True))

            registry = ToolRegistry()
            tool = Tool(
                name="numeric_tool",
                description="Returns a number",
                category=ToolCategory.MATH,
                parameters={},
                handler=lambda: 42,
            )
            registry.register_tool(tool)
            registry.set_role_permissions(
                "test_role",
                ToolPermissions(allowed_categories=[ToolCategory.MATH]),
            )

            result = registry.invoke("numeric_tool", role="test_role")
            assert result == 42
        finally:
            reset_features()


class TestREPLRedaction:
    """Integration test: verify REPL execute() redacts output."""

    def test_repl_stdout_redacted(self):
        from src.features import Features, set_features, reset_features
        from src.repl_environment.environment import REPLEnvironment

        try:
            set_features(Features(credential_redaction=True, repl=True))

            env = REPLEnvironment(context="test context")
            result = env.execute('print("key = AKIAIOSFODNN7EXAMPLE")')
            assert "AKIAIOSFODNN7EXAMPLE" not in result.output
            assert "[REDACTED:aws_access_key]" in result.output
        finally:
            reset_features()

    def test_repl_clean_output_unchanged(self):
        from src.features import Features, set_features, reset_features
        from src.repl_environment.environment import REPLEnvironment

        try:
            set_features(Features(credential_redaction=True, repl=True))

            env = REPLEnvironment(context="test context")
            result = env.execute('print("Hello, world!")')
            assert "Hello, world!" in result.output
        finally:
            reset_features()

    def test_repl_structured_mode_redacted(self):
        from src.features import Features, set_features, reset_features
        from src.repl_environment.environment import REPLEnvironment

        try:
            set_features(Features(credential_redaction=True, repl=True))

            env = REPLEnvironment(context="test context", structured_mode=True)
            result = env.execute('print("token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij1234")')
            assert "ghp_ABCDEFGH" not in result.output
            assert "[REDACTED:github_pat]" in result.output
        finally:
            reset_features()

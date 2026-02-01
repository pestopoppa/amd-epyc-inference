#!/usr/bin/env python3
"""Unit tests for src/roles.py."""

import pytest

from src.roles import (
    Role,
    Tier,
    chain_name_to_role,
    get_escalation_chain,
    get_tier,
    role_to_chain_name,
)


class TestTierEnum:
    """Test Tier enum."""

    def test_tier_values(self):
        """Test Tier enum values."""
        assert Tier.A == "A"
        assert Tier.B == "B"
        assert Tier.C == "C"
        assert Tier.D == "D"


class TestRoleEnum:
    """Test Role enum."""

    def test_role_values(self):
        """Test Role enum has expected values."""
        assert Role.FRONTDOOR == "frontdoor"
        assert Role.CODER_PRIMARY == "coder_primary"
        assert Role.WORKER_GENERAL == "worker_general"
        assert Role.DRAFT_CODER == "draft_coder"

    def test_role_str_conversion(self):
        """Test Role.__str__ returns value."""
        assert str(Role.CODER_PRIMARY) == "coder_primary"
        assert str(Role.WORKER_MATH) == "worker_math"

    def test_role_is_valid(self):
        """Test Role.is_valid() validates strings."""
        assert Role.is_valid("frontdoor") is True
        assert Role.is_valid("coder_primary") is True
        assert Role.is_valid("invalid_role") is False
        assert Role.is_valid("") is False

    def test_role_from_string_valid(self):
        """Test Role.from_string() with valid roles."""
        role = Role.from_string("coder_primary")
        assert role == Role.CODER_PRIMARY

        role2 = Role.from_string("worker_math")
        assert role2 == Role.WORKER_MATH

    def test_role_from_string_invalid(self):
        """Test Role.from_string() with invalid role returns default."""
        role = Role.from_string("invalid_role")
        assert role is None

        role2 = Role.from_string("invalid", default=Role.WORKER_GENERAL)
        assert role2 == Role.WORKER_GENERAL

    def test_role_tier_property(self):
        """Test Role.tier property."""
        assert Role.FRONTDOOR.tier == Tier.A
        assert Role.CODER_PRIMARY.tier == Tier.B
        assert Role.WORKER_GENERAL.tier == Tier.C
        assert Role.DRAFT_CODER.tier == Tier.D

    def test_role_is_specialist(self):
        """Test Role.is_specialist property."""
        assert Role.CODER_PRIMARY.is_specialist is True
        assert Role.ARCHITECT_GENERAL.is_specialist is True
        assert Role.WORKER_GENERAL.is_specialist is False
        assert Role.FRONTDOOR.is_specialist is False

    def test_role_is_worker(self):
        """Test Role.is_worker property."""
        assert Role.WORKER_GENERAL.is_worker is True
        assert Role.WORKER_MATH.is_worker is True
        assert Role.CODER_PRIMARY.is_worker is False

    def test_role_is_draft(self):
        """Test Role.is_draft property."""
        assert Role.DRAFT_CODER.is_draft is True
        assert Role.DRAFT_GENERAL.is_draft is True
        assert Role.WORKER_GENERAL.is_draft is False


class TestEscalationChain:
    """Test escalation chain logic."""

    def test_worker_escalates_to_coder(self):
        """Test worker roles escalate to coder."""
        assert Role.WORKER_GENERAL.escalates_to() == Role.CODER_PRIMARY
        assert Role.WORKER_MATH.escalates_to() == Role.CODER_PRIMARY
        assert Role.WORKER_SUMMARIZE.escalates_to() == Role.CODER_PRIMARY

    def test_frontdoor_escalates_to_coder(self):
        """Test frontdoor escalates to coder."""
        assert Role.FRONTDOOR.escalates_to() == Role.CODER_PRIMARY

    def test_coder_escalates_to_architect(self):
        """Test coder roles escalate to architect."""
        assert Role.CODER_PRIMARY.escalates_to() == Role.ARCHITECT_GENERAL
        assert Role.CODER_ESCALATION.escalates_to() == Role.ARCHITECT_CODING

    def test_ingest_escalates_to_architect(self):
        """Test ingest escalates to architect."""
        assert Role.INGEST_LONG_CONTEXT.escalates_to() == Role.ARCHITECT_GENERAL

    def test_architect_no_escalation(self):
        """Test architects have no escalation (top of chain)."""
        assert Role.ARCHITECT_GENERAL.escalates_to() is None
        assert Role.ARCHITECT_CODING.escalates_to() is None

    def test_draft_no_escalation(self):
        """Test draft models don't escalate."""
        assert Role.DRAFT_CODER.escalates_to() is None
        assert Role.DRAFT_GENERAL.escalates_to() is None


class TestGetTier:
    """Test get_tier() function."""

    def test_get_tier_from_role_enum(self):
        """Test get_tier with Role enum."""
        assert get_tier(Role.FRONTDOOR) == Tier.A
        assert get_tier(Role.CODER_PRIMARY) == Tier.B
        assert get_tier(Role.WORKER_GENERAL) == Tier.C
        assert get_tier(Role.DRAFT_CODER) == Tier.D

    def test_get_tier_from_string(self):
        """Test get_tier with string role."""
        assert get_tier("frontdoor") == Tier.A
        assert get_tier("coder_primary") == Tier.B
        assert get_tier("worker_math") == Tier.C
        assert get_tier("draft_general") == Tier.D

    def test_get_tier_unknown_defaults_to_c(self):
        """Test get_tier returns Tier.C for unknown roles."""
        assert get_tier("unknown_role") == Tier.C


class TestGetEscalationChain:
    """Test get_escalation_chain() function."""

    def test_worker_escalation_chain(self):
        """Test escalation chain from worker."""
        chain = get_escalation_chain(Role.WORKER_GENERAL)
        assert len(chain) == 3
        assert chain[0] == Role.WORKER_GENERAL
        assert chain[1] == Role.CODER_PRIMARY
        assert chain[2] == Role.ARCHITECT_GENERAL

    def test_frontdoor_escalation_chain(self):
        """Test escalation chain from frontdoor."""
        chain = get_escalation_chain(Role.FRONTDOOR)
        assert len(chain) == 3
        assert chain[0] == Role.FRONTDOOR
        assert chain[1] == Role.CODER_PRIMARY
        assert chain[2] == Role.ARCHITECT_GENERAL

    def test_coder_escalation_chain(self):
        """Test escalation chain from coder."""
        chain = get_escalation_chain(Role.CODER_PRIMARY)
        assert len(chain) == 2
        assert chain[0] == Role.CODER_PRIMARY
        assert chain[1] == Role.ARCHITECT_GENERAL

    def test_architect_escalation_chain(self):
        """Test escalation chain from architect (terminal)."""
        chain = get_escalation_chain(Role.ARCHITECT_GENERAL)
        assert len(chain) == 1
        assert chain[0] == Role.ARCHITECT_GENERAL

    def test_escalation_chain_from_string(self):
        """Test get_escalation_chain with string role."""
        chain = get_escalation_chain("worker_math")
        assert len(chain) == 3
        assert chain[0] == Role.WORKER_MATH

    def test_escalation_chain_unknown_role(self):
        """Test get_escalation_chain with unknown role returns empty."""
        chain = get_escalation_chain("unknown_role")
        assert chain == []


class TestChainNameMapping:
    """Test chain name to role mapping."""

    def test_chain_name_to_role(self):
        """Test chain_name_to_role conversion."""
        assert chain_name_to_role("worker") == Role.WORKER_GENERAL
        assert chain_name_to_role("coder") == Role.CODER_PRIMARY
        assert chain_name_to_role("architect") == Role.ARCHITECT_GENERAL
        assert chain_name_to_role("ingest") == Role.INGEST_LONG_CONTEXT
        assert chain_name_to_role("frontdoor") == Role.FRONTDOOR

    def test_chain_name_to_role_unknown(self):
        """Test chain_name_to_role returns None for unknown."""
        assert chain_name_to_role("unknown") is None

    def test_role_to_chain_name(self):
        """Test role_to_chain_name conversion."""
        assert role_to_chain_name(Role.WORKER_GENERAL) == "worker"
        assert role_to_chain_name(Role.CODER_PRIMARY) == "coder"
        assert role_to_chain_name(Role.ARCHITECT_GENERAL) == "architect"

    def test_role_to_chain_name_variants(self):
        """Test role_to_chain_name for variant roles."""
        assert role_to_chain_name(Role.WORKER_MATH) == "worker"
        assert role_to_chain_name(Role.CODER_ESCALATION) == "coder"
        assert role_to_chain_name(Role.ARCHITECT_CODING) == "architect"

    def test_role_to_chain_name_fallback(self):
        """Test role_to_chain_name returns value for unmapped roles."""
        # Thinking reasoning should map to coder
        assert role_to_chain_name(Role.THINKING_REASONING) == "coder"

"""Unit tests for Phase 4: 3-way routing evaluation."""

import sys

sys.path.insert(0, "scripts/benchmark")


class TestSeeding3WayTypes:
    """Tests for seeding_types.py 3-way constants."""

    def test_action_constants_defined(self):
        from seeding_types import (
            ACTION_SELF_DIRECT,
            ACTION_SELF_REPL,
            ACTION_ARCHITECT,
            ACTION_WORKER,
        )

        assert ACTION_SELF_DIRECT == "SELF:direct"
        assert ACTION_SELF_REPL == "SELF:repl"
        assert ACTION_ARCHITECT == "ARCHITECT"
        assert ACTION_WORKER == "WORKER"

    def test_three_way_actions_list(self):
        from seeding_types import THREE_WAY_ACTIONS

        assert len(THREE_WAY_ACTIONS) == 4
        assert "SELF:direct" in THREE_WAY_ACTIONS
        assert "SELF:repl" in THREE_WAY_ACTIONS
        assert "ARCHITECT" in THREE_WAY_ACTIONS
        assert "WORKER" in THREE_WAY_ACTIONS

    def test_cost_tiers_defined(self):
        from seeding_types import THREE_WAY_COST_TIER

        assert THREE_WAY_COST_TIER["SELF:direct"] == 2
        assert THREE_WAY_COST_TIER["SELF:repl"] == 2
        assert THREE_WAY_COST_TIER["ARCHITECT"] == 4
        assert THREE_WAY_COST_TIER["WORKER"] == 1


class TestSeeding3WayRewards:
    """Tests for seeding_rewards.py 3-way reward functions."""

    def test_success_reward_binary(self):
        from seeding_rewards import success_reward

        assert success_reward(True) == 1.0
        assert success_reward(False) == 0.0

    def test_tool_value_tools_helped(self):
        from seeding_rewards import compute_tool_value

        result = compute_tool_value(direct_passed=False, repl_passed=True)
        assert result["tools_helped"] is True
        assert result["tools_neutral"] is False
        assert result["tools_hurt"] is False
        assert result["tool_advantage"] == 1

    def test_tool_value_tools_neutral(self):
        from seeding_rewards import compute_tool_value

        result = compute_tool_value(direct_passed=True, repl_passed=True)
        assert result["tools_helped"] is False
        assert result["tools_neutral"] is True
        assert result["tools_hurt"] is False
        assert result["tool_advantage"] == 0

    def test_tool_value_tools_hurt(self):
        from seeding_rewards import compute_tool_value

        result = compute_tool_value(direct_passed=True, repl_passed=False)
        assert result["tools_helped"] is False
        assert result["tools_neutral"] is False
        assert result["tools_hurt"] is True
        assert result["tool_advantage"] == -1

    def test_compute_3way_rewards_all_pass(self):
        from seeding_rewards import compute_3way_rewards
        from seeding_types import RoleResult

        results = {
            "frontdoor:direct": RoleResult(
                role="frontdoor", mode="direct", answer="ok", passed=True, elapsed_seconds=1.0
            ),
            "frontdoor:repl": RoleResult(
                role="frontdoor", mode="repl", answer="ok", passed=True, elapsed_seconds=1.0
            ),
            "architect_general:delegated": RoleResult(
                role="architect_general", mode="delegated", answer="ok", passed=True, elapsed_seconds=1.0
            ),
        }

        rewards = compute_3way_rewards(results)
        assert rewards["SELF:direct"] == 1.0
        assert rewards["SELF:repl"] == 1.0
        assert rewards["ARCHITECT"] == 1.0

    def test_compute_3way_rewards_all_fail(self):
        from seeding_rewards import compute_3way_rewards
        from seeding_types import RoleResult

        results = {
            "frontdoor:direct": RoleResult(
                role="frontdoor", mode="direct", answer="wrong", passed=False, elapsed_seconds=1.0
            ),
            "frontdoor:repl": RoleResult(
                role="frontdoor", mode="repl", answer="wrong", passed=False, elapsed_seconds=1.0
            ),
            "architect_coding:delegated": RoleResult(
                role="architect_coding", mode="delegated", answer="wrong", passed=False, elapsed_seconds=1.0
            ),
        }

        rewards = compute_3way_rewards(results)
        assert rewards["SELF:direct"] == 0.0
        assert rewards["SELF:repl"] == 0.0
        assert rewards["ARCHITECT"] == 0.0


class TestHybridRouter3Way:
    """Tests for HybridRouter.route_3way method."""

    def test_route_3way_fallback_to_rules(self):
        """Test that route_3way falls back to rule-based routing."""
        from orchestration.repl_memory.retriever import (
            HybridRouter,
            RuleBasedRouter,
            TwoPhaseRetriever,
            RetrievalConfig,
        )
        from orchestration.repl_memory.episodic_store import EpisodicStore
        from orchestration.repl_memory.embedder import TaskEmbedder
        from unittest.mock import MagicMock

        # Create mock components
        mock_store = MagicMock(spec=EpisodicStore)
        mock_store.retrieve_by_similarity.return_value = []

        mock_embedder = MagicMock(spec=TaskEmbedder)
        mock_embedder.embed_task_ir.return_value = [0.0] * 896

        config = RetrievalConfig()
        retriever = TwoPhaseRetriever(mock_store, mock_embedder, config)

        rule_router = RuleBasedRouter(routing_hints=[])
        hybrid_router = HybridRouter(retriever, rule_router)

        # Test rule-based fallback
        task_ir = {"task_type": "chat", "objective": "Hello world"}
        action, strategy, confidence = hybrid_router.route_3way(task_ir)

        assert strategy == "rules"
        assert confidence == 0.5
        assert action in ["SELF:direct", "SELF:repl", "ARCHITECT"]

    def test_route_3way_architecture_task(self):
        """Test that architecture tasks route to ARCHITECT."""
        from orchestration.repl_memory.retriever import (
            HybridRouter,
            RuleBasedRouter,
            TwoPhaseRetriever,
            RetrievalConfig,
        )
        from orchestration.repl_memory.episodic_store import EpisodicStore
        from orchestration.repl_memory.embedder import TaskEmbedder
        from unittest.mock import MagicMock

        mock_store = MagicMock(spec=EpisodicStore)
        mock_store.retrieve_by_similarity.return_value = []

        mock_embedder = MagicMock(spec=TaskEmbedder)
        mock_embedder.embed_task_ir.return_value = [0.0] * 896

        config = RetrievalConfig()
        retriever = TwoPhaseRetriever(mock_store, mock_embedder, config)
        rule_router = RuleBasedRouter(routing_hints=[])
        hybrid_router = HybridRouter(retriever, rule_router)

        task_ir = {"task_type": "architecture", "objective": "Design a system"}
        action, strategy, confidence = hybrid_router.route_3way(task_ir)

        assert action == "ARCHITECT"
        assert strategy == "rules"

    def test_route_3way_file_task_routes_to_repl(self):
        """Test that file operations route to SELF:repl."""
        from orchestration.repl_memory.retriever import (
            HybridRouter,
            RuleBasedRouter,
            TwoPhaseRetriever,
            RetrievalConfig,
        )
        from orchestration.repl_memory.episodic_store import EpisodicStore
        from orchestration.repl_memory.embedder import TaskEmbedder
        from unittest.mock import MagicMock

        mock_store = MagicMock(spec=EpisodicStore)
        mock_store.retrieve_by_similarity.return_value = []

        mock_embedder = MagicMock(spec=TaskEmbedder)
        mock_embedder.embed_task_ir.return_value = [0.0] * 896

        config = RetrievalConfig()
        retriever = TwoPhaseRetriever(mock_store, mock_embedder, config)
        rule_router = RuleBasedRouter(routing_hints=[])
        hybrid_router = HybridRouter(retriever, rule_router)

        task_ir = {"task_type": "chat", "objective": "read the file config.yaml"}
        action, strategy, confidence = hybrid_router.route_3way(task_ir)

        assert action == "SELF:repl"
        assert strategy == "rules"


class TestInfraErrorClassification:
    """Tests for infrastructure error classification in seeding."""

    def test_classify_error_infra(self):
        from seed_specialist_routing import _classify_error

        assert _classify_error("ReadTimeout: timed out") == "infrastructure"
        assert _classify_error("HTTP 503 backend down") == "infrastructure"

    def test_classify_error_task_failure(self):
        from seed_specialist_routing import _classify_error

        assert _classify_error("model produced wrong answer") == "task_failure"
        assert _classify_error(None) == "none"


class TestChatRequestAllowDelegation:
    """Tests for ChatRequest.allow_delegation field."""

    def test_allow_delegation_default_none(self):
        from src.api.models import ChatRequest

        req = ChatRequest(prompt="test")
        assert req.allow_delegation is None

    def test_allow_delegation_true(self):
        from src.api.models import ChatRequest

        req = ChatRequest(prompt="test", allow_delegation=True)
        assert req.allow_delegation is True

    def test_allow_delegation_false(self):
        from src.api.models import ChatRequest

        req = ChatRequest(prompt="test", allow_delegation=False)
        assert req.allow_delegation is False

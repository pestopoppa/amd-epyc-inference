"""
Unit tests for SkillBank and SkillRetriever.

All tests use :memory: SQLite and temp directories — no live inference,
no model loading, no API calls.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── SkillBank Tests ──────────────────────────────────────────────────────


class TestSkill:
    """Tests for Skill dataclass."""

    def test_create_valid_skill(self):
        from orchestration.repl_memory.skill_bank import Skill

        skill = Skill(
            id="route_001",
            title="Prefer Coder for Refactoring",
            skill_type="routing",
            principle="Route refactoring tasks to coder_escalation.",
            when_to_apply="task_type is refactoring",
            teacher_model="claude-opus-4-6",
        )
        assert skill.id == "route_001"
        assert skill.skill_type == "routing"
        assert skill.confidence == 0.5
        assert skill.deprecated is False

    def test_invalid_skill_type_raises(self):
        from orchestration.repl_memory.skill_bank import Skill

        with pytest.raises(ValueError, match="Invalid skill_type"):
            Skill(
                id="bad_001",
                title="Bad Skill",
                skill_type="invalid",
                principle="...",
                when_to_apply="...",
                teacher_model="test",
            )

    def test_to_dict_roundtrip(self):
        from orchestration.repl_memory.skill_bank import Skill

        skill = Skill(
            id="gen_001",
            title="Test Skill",
            skill_type="general",
            principle="Do the thing.",
            when_to_apply="Always",
            task_types=["code_generation", "debugging"],
            source_trajectory_ids=["traj_1", "traj_2"],
            source_outcome="success",
            confidence=0.8,
            teacher_model="test-model",
        )
        d = skill.to_dict()
        restored = Skill.from_dict(d)
        assert restored.id == skill.id
        assert restored.task_types == ["code_generation", "debugging"]
        assert restored.source_trajectory_ids == ["traj_1", "traj_2"]
        assert restored.confidence == 0.8

    def test_from_dict_handles_json_strings(self):
        from orchestration.repl_memory.skill_bank import Skill

        d = {
            "id": "route_002",
            "title": "Test",
            "skill_type": "routing",
            "principle": "...",
            "when_to_apply": "...",
            "task_types": '["code"]',
            "source_trajectory_ids": '["t1"]',
            "source_outcome": "success",
            "deprecated": 0,
            "created_at": "2026-02-14T10:00:00",
            "updated_at": "2026-02-14T10:00:00",
            "teacher_model": "test",
        }
        skill = Skill.from_dict(d)
        assert skill.task_types == ["code"]
        assert skill.deprecated is False


class TestSkillBank:
    """Tests for SkillBank SQLite + FAISS store."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def bank(self, temp_dir):
        from orchestration.repl_memory.skill_bank import SkillBank

        return SkillBank(
            db_path=temp_dir / "skills.db",
            faiss_path=temp_dir,
            embedding_dim=128,  # Small for testing
        )

    @pytest.fixture
    def sample_skill(self):
        from orchestration.repl_memory.skill_bank import Skill

        return Skill(
            id="route_001",
            title="Prefer Coder for Refactoring",
            skill_type="routing",
            principle="Route refactoring tasks to coder_escalation (port 8081).",
            when_to_apply="task_type is refactoring or code_review",
            task_types=["refactoring", "code_review"],
            source_trajectory_ids=["traj_001", "traj_002"],
            source_outcome="success",
            confidence=0.75,
            teacher_model="claude-opus-4-6",
        )

    def test_create_empty_bank(self, bank):
        assert bank.count() == 0
        stats = bank.get_stats()
        assert stats["total"] == 0

    def test_store_and_retrieve(self, bank, sample_skill):
        bank.store(sample_skill)
        retrieved = bank.get_by_id("route_001")
        assert retrieved is not None
        assert retrieved.title == "Prefer Coder for Refactoring"
        assert retrieved.task_types == ["refactoring", "code_review"]
        assert retrieved.confidence == 0.75

    def test_store_with_embedding(self, bank, sample_skill):
        embedding = np.random.randn(128).astype(np.float32)
        bank.store(sample_skill, embedding=embedding)
        retrieved = bank.get_by_id("route_001")
        assert retrieved.embedding_idx == 0

    def test_count(self, bank):
        from orchestration.repl_memory.skill_bank import Skill

        for i in range(5):
            skill = Skill(
                id=f"route_{i:03d}",
                title=f"Skill {i}",
                skill_type="routing",
                principle=f"Do thing {i}.",
                when_to_apply="Always",
                teacher_model="test",
            )
            bank.store(skill)
        assert bank.count() == 5

    def test_count_excludes_deprecated(self, bank):
        from orchestration.repl_memory.skill_bank import Skill

        skill = Skill(
            id="route_001",
            title="Active",
            skill_type="routing",
            principle="...",
            when_to_apply="...",
            teacher_model="test",
        )
        bank.store(skill)
        deprecated = Skill(
            id="route_002",
            title="Deprecated",
            skill_type="routing",
            principle="...",
            when_to_apply="...",
            deprecated=True,
            teacher_model="test",
        )
        bank.store(deprecated)
        assert bank.count(include_deprecated=False) == 1
        assert bank.count(include_deprecated=True) == 2

    def test_update_fields(self, bank, sample_skill):
        bank.store(sample_skill)
        updated = bank.update("route_001", confidence=0.9, revision=2)
        assert updated is True
        skill = bank.get_by_id("route_001")
        assert skill.confidence == 0.9
        assert skill.revision == 2

    def test_update_nonexistent_returns_false(self, bank):
        result = bank.update("nonexistent", confidence=0.9)
        assert result is False

    def test_update_disallowed_field_raises(self, bank, sample_skill):
        bank.store(sample_skill)
        with pytest.raises(ValueError, match="Cannot update field"):
            bank.update("route_001", id="hacked")

    def test_deprecate(self, bank, sample_skill):
        bank.store(sample_skill)
        bank.deprecate("route_001")
        skill = bank.get_by_id("route_001")
        assert skill.deprecated is True

    def test_increment_retrieval(self, bank, sample_skill):
        bank.store(sample_skill)
        bank.increment_retrieval(["route_001", "route_001"])
        skill = bank.get_by_id("route_001")
        assert skill.retrieval_count == 2

    def test_get_skills_by_type(self, bank):
        from orchestration.repl_memory.skill_bank import Skill

        for i, stype in enumerate(["general", "routing", "routing", "failure_lesson"]):
            skill = Skill(
                id=f"skill_{i:03d}",
                title=f"Skill {i}",
                skill_type=stype,
                principle=f"Principle {i}.",
                when_to_apply="Always",
                teacher_model="test",
            )
            bank.store(skill)

        routing = bank.get_skills(skill_type="routing")
        assert len(routing) == 2
        general = bank.get_skills(skill_type="general")
        assert len(general) == 1

    def test_get_skills_by_confidence(self, bank):
        from orchestration.repl_memory.skill_bank import Skill

        for i, conf in enumerate([0.2, 0.5, 0.8, 0.95]):
            skill = Skill(
                id=f"skill_{i:03d}",
                title=f"Skill {i}",
                skill_type="routing",
                principle=f"Principle {i}.",
                when_to_apply="Always",
                confidence=conf,
                teacher_model="test",
            )
            bank.store(skill)

        high = bank.get_skills(min_confidence=0.7)
        assert len(high) == 2
        low = bank.get_skills(max_confidence=0.3)
        assert len(low) == 1

    def test_get_skills_by_task_type(self, bank):
        from orchestration.repl_memory.skill_bank import Skill

        skill_code = Skill(
            id="route_001",
            title="Code Skill",
            skill_type="routing",
            principle="...",
            when_to_apply="...",
            task_types=["code_generation"],
            teacher_model="test",
        )
        skill_general = Skill(
            id="gen_001",
            title="General Skill",
            skill_type="general",
            principle="...",
            when_to_apply="...",
            task_types=["*"],
            teacher_model="test",
        )
        bank.store(skill_code)
        bank.store(skill_general)

        results = bank.get_skills(task_type="code_generation")
        assert len(results) == 2  # Code skill + general (matches *)

    def test_search_by_embedding(self, bank):
        from orchestration.repl_memory.skill_bank import Skill

        # Store skills with embeddings
        base_vec = np.random.randn(128).astype(np.float32)
        for i in range(5):
            skill = Skill(
                id=f"route_{i:03d}",
                title=f"Skill {i}",
                skill_type="routing",
                principle=f"Principle {i}.",
                when_to_apply="Always",
                teacher_model="test",
            )
            # Create embeddings with decreasing similarity to base
            noise = np.random.randn(128).astype(np.float32) * (i * 0.5)
            emb = (base_vec + noise).astype(np.float32)
            bank.store(skill, embedding=emb)

        # Search with the base vector
        results = bank.search_by_embedding(base_vec, k=3)
        assert len(results) <= 3
        # Results should be sorted by similarity descending
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

    def test_search_excludes_deprecated(self, bank):
        from orchestration.repl_memory.skill_bank import Skill

        emb = np.random.randn(128).astype(np.float32)
        skill = Skill(
            id="route_001",
            title="Deprecated Skill",
            skill_type="routing",
            principle="...",
            when_to_apply="...",
            deprecated=True,
            teacher_model="test",
        )
        bank.store(skill, embedding=emb)

        results = bank.search_by_embedding(emb, k=5, exclude_deprecated=True)
        assert len(results) == 0

        results = bank.search_by_embedding(emb, k=5, exclude_deprecated=False)
        assert len(results) == 1

    def test_find_duplicates(self, bank):
        from orchestration.repl_memory.skill_bank import Skill

        emb = np.random.randn(128).astype(np.float32)
        skill = Skill(
            id="route_001",
            title="Original",
            skill_type="routing",
            principle="...",
            when_to_apply="...",
            teacher_model="test",
        )
        bank.store(skill, embedding=emb)

        # An almost-identical embedding should be found as duplicate
        near_dup = emb + np.random.randn(128).astype(np.float32) * 0.01
        duplicates = bank.find_duplicates(near_dup, threshold=0.8)
        assert len(duplicates) >= 1
        assert duplicates[0][0].id == "route_001"

    def test_get_stats(self, bank):
        from orchestration.repl_memory.skill_bank import Skill

        for stype in ["general", "routing", "escalation", "failure_lesson"]:
            skill = Skill(
                id=f"{stype}_001",
                title=f"Test {stype}",
                skill_type=stype,
                principle="...",
                when_to_apply="...",
                confidence=0.7,
                teacher_model="test",
            )
            bank.store(skill)

        stats = bank.get_stats()
        assert stats["total"] == 4
        assert stats["general"] == 1
        assert stats["routing"] == 1
        assert stats["escalation"] == 1
        assert stats["failure_lesson"] == 1
        assert stats["deprecated"] == 0
        assert abs(stats["avg_confidence"] - 0.7) < 0.01

    def test_generate_id_with_seq(self):
        from orchestration.repl_memory.skill_bank import SkillBank

        assert SkillBank.generate_id("general", 1) == "gen_001"
        assert SkillBank.generate_id("routing", 42) == "route_042"
        assert SkillBank.generate_id("escalation", 3) == "esc_003"
        assert SkillBank.generate_id("failure_lesson", 7) == "fail_007"

    def test_generate_id_without_seq(self):
        from orchestration.repl_memory.skill_bank import SkillBank

        id1 = SkillBank.generate_id("routing")
        id2 = SkillBank.generate_id("routing")
        assert id1.startswith("route_")
        assert id2.startswith("route_")
        assert id1 != id2  # UUID-based, should differ

    def test_upsert_overwrites(self, bank, sample_skill):
        bank.store(sample_skill)
        # Modify and re-store
        sample_skill.confidence = 0.99
        sample_skill.revision = 2
        bank.store(sample_skill)
        # Should have 1 skill, not 2
        assert bank.count() == 1
        skill = bank.get_by_id("route_001")
        assert skill.confidence == 0.99

    def test_close(self, bank, sample_skill):
        bank.store(sample_skill)
        bank.close()
        # Should not raise


# ── SkillRetriever Tests ─────────────────────────────────────────────────


class TestSkillRetriever:
    """Tests for SkillRetriever."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def bank(self, temp_dir):
        from orchestration.repl_memory.skill_bank import SkillBank

        return SkillBank(
            db_path=temp_dir / "skills.db",
            faiss_path=temp_dir,
            embedding_dim=128,
        )

    @pytest.fixture
    def populated_bank(self, bank):
        """Bank with a mix of general and task-specific skills + embeddings."""
        from orchestration.repl_memory.skill_bank import Skill

        skills = [
            Skill(
                id="gen_001",
                title="Always Check Constraints",
                skill_type="general",
                principle="Verify model constraints before routing.",
                when_to_apply="Every routing decision",
                task_types=["*"],
                confidence=0.9,
                teacher_model="test",
            ),
            Skill(
                id="gen_002",
                title="SSM Cannot Speculate",
                skill_type="general",
                principle="Never use speculative decoding with SSM models.",
                when_to_apply="Routing involves ingest_long_context",
                task_types=["*"],
                confidence=0.95,
                teacher_model="test",
            ),
            Skill(
                id="route_001",
                title="Coder for Refactoring",
                skill_type="routing",
                principle="Route refactoring tasks to coder_escalation.",
                when_to_apply="task_type is refactoring",
                task_types=["refactoring"],
                confidence=0.8,
                teacher_model="test",
            ),
            Skill(
                id="route_002",
                title="Worker for Simple Debug",
                skill_type="routing",
                principle="Use worker_explore for single-file debugging.",
                when_to_apply="task_type is debugging, single file",
                task_types=["debugging"],
                confidence=0.7,
                teacher_model="test",
            ),
            Skill(
                id="fail_001",
                title="Avoid Prompt Lookup Novel Code",
                skill_type="failure_lesson",
                principle="FAILURE: Prompt lookup returns 0 on novel code.",
                when_to_apply="code_generation without existing context",
                task_types=["code_generation"],
                confidence=0.85,
                teacher_model="test",
            ),
        ]

        np.random.seed(42)
        for skill in skills:
            emb = np.random.randn(128).astype(np.float32)
            bank.store(skill, embedding=emb)

        return bank

    @pytest.fixture
    def retriever(self, populated_bank):
        from orchestration.repl_memory.skill_retriever import (
            SkillRetriever,
            SkillRetrievalConfig,
        )

        config = SkillRetrievalConfig(
            general_skills_max=6,
            task_specific_k=3,
            min_similarity=0.0,  # Low threshold for test with random embeddings
            min_confidence=0.3,
            max_prompt_tokens=2000,
        )
        return SkillRetriever(populated_bank, config=config)

    def test_retrieve_includes_general_skills(self, retriever):
        query = np.random.randn(128).astype(np.float32)
        results = retriever.retrieve_for_task(query, task_type="refactoring")
        general = [r for r in results if r.source == "general"]
        assert len(general) == 2  # gen_001 and gen_002

    def test_retrieve_includes_task_specific(self, retriever):
        query = np.random.randn(128).astype(np.float32)
        results = retriever.retrieve_for_task(query, task_type="refactoring")
        specific = [r for r in results if r.source == "task_specific"]
        assert len(specific) >= 1

    def test_retrieve_respects_task_type_filter(self, retriever):
        query = np.random.randn(128).astype(np.float32)
        results = retriever.retrieve_for_task(query, task_type="refactoring")
        for r in results:
            if r.source == "task_specific":
                assert (
                    "refactoring" in r.skill.task_types
                    or "*" in r.skill.task_types
                )

    def test_retrieve_no_duplicates(self, retriever):
        query = np.random.randn(128).astype(np.float32)
        results = retriever.retrieve_for_task(query)
        ids = [r.skill.id for r in results]
        assert len(ids) == len(set(ids))

    def test_format_for_prompt_empty(self, retriever):
        output = retriever.format_for_prompt([])
        assert output == ""

    def test_format_for_prompt_has_sections(self, retriever):
        query = np.random.randn(128).astype(np.float32)
        results = retriever.retrieve_for_task(query)
        output = retriever.format_for_prompt(results)
        assert "## Learned Routing Skills" in output
        assert "### General Principles" in output

    def test_format_for_prompt_contains_skill_titles(self, retriever):
        query = np.random.randn(128).astype(np.float32)
        results = retriever.retrieve_for_task(query)
        output = retriever.format_for_prompt(results)
        assert "Always Check Constraints" in output
        assert "SSM Cannot Speculate" in output

    def test_format_for_prompt_respects_token_budget(self, populated_bank):
        from orchestration.repl_memory.skill_retriever import (
            SkillRetriever,
            SkillRetrievalConfig,
        )

        config = SkillRetrievalConfig(max_prompt_tokens=50, min_similarity=0.0)
        retriever = SkillRetriever(populated_bank, config=config)
        query = np.random.randn(128).astype(np.float32)
        results = retriever.retrieve_for_task(query)
        output = retriever.format_for_prompt(results)
        # With 50 tokens (~200 chars), output should be truncated
        assert len(output) < 500

    def test_retrieval_increments_count(self, populated_bank):
        from orchestration.repl_memory.skill_retriever import (
            SkillRetriever,
            SkillRetrievalConfig,
        )

        config = SkillRetrievalConfig(min_similarity=0.0, min_confidence=0.0)
        retriever = SkillRetriever(populated_bank, config=config)
        query = np.random.randn(128).astype(np.float32)

        # Initial retrieval count is 0
        before = populated_bank.get_by_id("gen_001")
        assert before.retrieval_count == 0

        retriever.retrieve_for_task(query)

        after = populated_bank.get_by_id("gen_001")
        assert after.retrieval_count == 1

    def test_retrieve_with_none_task_type(self, retriever):
        """Should still work without task_type filter."""
        query = np.random.randn(128).astype(np.float32)
        results = retriever.retrieve_for_task(query, task_type=None)
        assert len(results) >= 2  # At least the general skills


# ── FAISS Filename Customization Tests ───────────────────────────────────


class TestFAISSCustomFilenames:
    """Test that FAISSEmbeddingStore supports custom filenames."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_custom_filenames(self, temp_dir):
        from orchestration.repl_memory.faiss_store import FAISSEmbeddingStore

        store = FAISSEmbeddingStore(
            path=temp_dir,
            dim=64,
            index_filename="skill_embeddings.faiss",
            id_map_filename="skill_id_map.npy",
        )
        emb = np.random.randn(64).astype(np.float32)
        store.add("skill_001", emb)
        store.save()

        assert (temp_dir / "skill_embeddings.faiss").exists()
        assert (temp_dir / "skill_id_map.npy").exists()
        # Default files should NOT exist
        assert not (temp_dir / "embeddings.faiss").exists()

    def test_coexistence_with_default(self, temp_dir):
        """Two stores in the same directory with different filenames."""
        from orchestration.repl_memory.faiss_store import FAISSEmbeddingStore

        default_store = FAISSEmbeddingStore(path=temp_dir, dim=64)
        skill_store = FAISSEmbeddingStore(
            path=temp_dir,
            dim=64,
            index_filename="skill_embeddings.faiss",
            id_map_filename="skill_id_map.npy",
        )

        default_store.add("mem_001", np.random.randn(64).astype(np.float32))
        skill_store.add("skill_001", np.random.randn(64).astype(np.float32))

        default_store.save()
        skill_store.save()

        assert default_store.count == 1
        assert skill_store.count == 1

        # Reload and verify isolation
        default_reloaded = FAISSEmbeddingStore(path=temp_dir, dim=64)
        skill_reloaded = FAISSEmbeddingStore(
            path=temp_dir,
            dim=64,
            index_filename="skill_embeddings.faiss",
            id_map_filename="skill_id_map.npy",
        )
        assert default_reloaded.count == 1
        assert skill_reloaded.count == 1
        assert default_reloaded.id_map[0] == "mem_001"
        assert skill_reloaded.id_map[0] == "skill_001"

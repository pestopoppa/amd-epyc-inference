"""
Unit tests for the distillation pipeline.

All tests use MockTeacher — no live inference, no API calls.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest


# ── Response Parsing Tests ───────────────────────────────────────────────


class TestParseSkillsFromResponse:
    """Tests for parse_skills_from_response()."""

    def test_parse_fenced_json_array(self):
        from orchestration.repl_memory.distillation.teachers import (
            parse_skills_from_response,
        )

        response = '''Here are the skills:

```json
[
  {"title": "Skill A", "principle": "Do A."},
  {"title": "Skill B", "principle": "Do B."}
]
```
'''
        skills = parse_skills_from_response(response)
        assert len(skills) == 2
        assert skills[0]["title"] == "Skill A"
        assert skills[1]["title"] == "Skill B"

    def test_parse_fenced_json_single_object(self):
        from orchestration.repl_memory.distillation.teachers import (
            parse_skills_from_response,
        )

        response = '''```json
{"title": "Single Skill", "principle": "Do it."}
```'''
        skills = parse_skills_from_response(response)
        assert len(skills) == 1
        assert skills[0]["title"] == "Single Skill"

    def test_parse_bare_json_fallback(self):
        from orchestration.repl_memory.distillation.teachers import (
            parse_skills_from_response,
        )

        response = '''
Here is a skill:
{"title": "Bare Skill", "principle": "Do bare."}
And another:
{"title": "Bare Skill 2", "principle": "Do bare 2."}
'''
        skills = parse_skills_from_response(response)
        assert len(skills) == 2

    def test_parse_no_skills(self):
        from orchestration.repl_memory.distillation.teachers import (
            parse_skills_from_response,
        )

        response = "No useful patterns found in these trajectories."
        skills = parse_skills_from_response(response)
        assert len(skills) == 0

    def test_parse_invalid_json_skipped(self):
        from orchestration.repl_memory.distillation.teachers import (
            parse_skills_from_response,
        )

        response = '''```json
{invalid json here}
```'''
        skills = parse_skills_from_response(response)
        assert len(skills) == 0

    def test_parse_mixed_valid_invalid(self):
        from orchestration.repl_memory.distillation.teachers import (
            parse_skills_from_response,
        )

        response = '''```json
[
  {"title": "Good Skill", "principle": "Works."},
  "this is not a dict"
]
```'''
        # The array parse includes the string, but we only filter dicts in the pipeline
        skills = parse_skills_from_response(response)
        assert len(skills) >= 1


# ── Prompt Building Tests ────────────────────────────────────────────────


class TestPromptBuilding:
    """Tests for prompt template rendering."""

    def test_build_success_prompt(self):
        from orchestration.repl_memory.distillation.prompts import build_success_prompt

        trajectories = [
            {"task_id": "t1", "task_type": "code_generation", "outcome": "success"},
        ]
        prompt = build_success_prompt(trajectories)
        assert "Distill Routing Skills" in prompt
        assert "code_generation" in prompt
        assert "task_id" in prompt

    def test_build_failure_prompt(self):
        from orchestration.repl_memory.distillation.prompts import build_failure_prompt

        trajectories = [
            {"task_id": "t2", "task_type": "debugging", "outcome": "failure"},
        ]
        prompt = build_failure_prompt(
            trajectories, failure_graph_summary="3 known timeout failures"
        )
        assert "Failure Lessons" in prompt
        assert "3 known timeout failures" in prompt
        assert "FAILURE POINT" in prompt

    def test_build_escalation_prompt(self):
        from orchestration.repl_memory.distillation.prompts import (
            build_escalation_prompt,
        )

        trajectories = [
            {"task_id": "t3", "task_type": "refactoring", "escalations": ["coder->architect"]},
        ]
        prompt = build_escalation_prompt(trajectories)
        assert "Escalation Patterns" in prompt
        assert "TRANSFERABLE reasoning" in prompt


# ── MockTeacher Tests ────────────────────────────────────────────────────


class TestMockTeacher:
    """Tests for MockTeacher."""

    def test_mock_returns_configured_responses(self):
        from orchestration.repl_memory.distillation.teachers import MockTeacher

        teacher = MockTeacher(responses=["response_1", "response_2"])
        assert asyncio.run(teacher.distill("prompt 1")) == "response_1"
        assert asyncio.run(teacher.distill("prompt 2")) == "response_2"

    def test_mock_records_prompts(self):
        from orchestration.repl_memory.distillation.teachers import MockTeacher

        teacher = MockTeacher(responses=["ok"])
        asyncio.run(teacher.distill("my prompt"))
        assert len(teacher.prompts) == 1
        assert "my prompt" in teacher.prompts[0]

    def test_mock_model_id(self):
        from orchestration.repl_memory.distillation.teachers import MockTeacher

        teacher = MockTeacher(model_id="test-v1")
        assert teacher.model_id == "test-v1"


# ── Pipeline Tests ───────────────────────────────────────────────────────


class TestDistillationPipeline:
    """Tests for the full distillation pipeline."""

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

    def _make_mock_response(self, skills):
        """Build a fenced JSON response from a list of skill dicts."""
        return f"```json\n{json.dumps(skills, indent=2)}\n```"

    def test_pipeline_success_distillation(self, bank):
        from orchestration.repl_memory.distillation.teachers import MockTeacher
        from orchestration.repl_memory.distillation.pipeline import DistillationPipeline

        teacher_response = self._make_mock_response([
            {
                "title": "Prefer Coder for Refactoring",
                "skill_type": "routing",
                "principle": "Route refactoring to coder_escalation.",
                "when_to_apply": "task_type is refactoring",
                "task_types": ["refactoring"],
                "source_outcome": "success",
            }
        ])
        teacher = MockTeacher(responses=[teacher_response])
        pipeline = DistillationPipeline(teacher=teacher, skill_bank=bank)

        trajectories = [
            {
                "task_id": f"t{i}",
                "task_type": "refactoring",
                "objective": "Refactor module",
                "routing_decision": "coder_escalation",
                "outcome": "success",
            }
            for i in range(5)
        ]

        report = asyncio.run(pipeline.run(trajectories))

        assert report.total_trajectories == 5
        assert report.success_trajectories == 5
        assert report.skills_proposed == 1
        assert report.skills_stored == 1
        assert report.errors == []
        assert bank.count() == 1

        skill = bank.get_skills()[0]
        assert skill.title == "Prefer Coder for Refactoring"
        assert skill.teacher_model == "mock-teacher"

    def test_pipeline_failure_distillation(self, bank):
        from orchestration.repl_memory.distillation.teachers import MockTeacher
        from orchestration.repl_memory.distillation.pipeline import DistillationPipeline

        teacher_response = self._make_mock_response([
            {
                "title": "Avoid Prompt Lookup Novel Code",
                "skill_type": "failure_lesson",
                "principle": "FAILURE POINT: Prompt lookup returns 0 tokens. PREVENTION: Use spec decode.",
                "when_to_apply": "code_generation without existing context",
                "task_types": ["code_generation"],
                "source_outcome": "failure",
            }
        ])
        teacher = MockTeacher(responses=[teacher_response])
        pipeline = DistillationPipeline(teacher=teacher, skill_bank=bank)

        trajectories = [
            {
                "task_id": "t1",
                "task_type": "code_generation",
                "outcome": "failure",
            },
        ]

        report = asyncio.run(pipeline.run(trajectories))

        assert report.failure_trajectories == 1
        assert report.skills_stored == 1
        skill = bank.get_skills(skill_type="failure_lesson")[0]
        assert "FAILURE POINT" in skill.principle

    def test_pipeline_escalation_distillation(self, bank):
        from orchestration.repl_memory.distillation.teachers import MockTeacher
        from orchestration.repl_memory.distillation.pipeline import DistillationPipeline

        teacher_response = self._make_mock_response([
            {
                "title": "Debug Locally Before Escalating",
                "skill_type": "escalation",
                "principle": "Try worker_explore with REPL before escalating to architect.",
                "when_to_apply": "debugging with clear error trace",
                "task_types": ["debugging"],
                "source_outcome": "success",
            }
        ])
        teacher = MockTeacher(responses=[teacher_response])
        pipeline = DistillationPipeline(teacher=teacher, skill_bank=bank)

        trajectories = [
            {
                "task_id": "t1",
                "task_type": "debugging",
                "outcome": "success",
                "escalations": ["worker->coder"],
            },
        ]

        report = asyncio.run(pipeline.run(trajectories))

        assert report.escalation_trajectories == 1
        assert report.skills_stored == 1

    def test_pipeline_mixed_trajectories(self, bank):
        from orchestration.repl_memory.distillation.teachers import MockTeacher
        from orchestration.repl_memory.distillation.pipeline import DistillationPipeline

        success_resp = self._make_mock_response([
            {"title": "S1", "skill_type": "routing", "principle": "P1",
             "when_to_apply": "W1", "task_types": ["*"], "source_outcome": "success"}
        ])
        failure_resp = self._make_mock_response([
            {"title": "F1", "skill_type": "failure_lesson", "principle": "P2",
             "when_to_apply": "W2", "task_types": ["*"], "source_outcome": "failure"}
        ])
        escalation_resp = self._make_mock_response([
            {"title": "E1", "skill_type": "escalation", "principle": "P3",
             "when_to_apply": "W3", "task_types": ["*"], "source_outcome": "success"}
        ])

        teacher = MockTeacher(responses=[success_resp, failure_resp, escalation_resp])
        pipeline = DistillationPipeline(teacher=teacher, skill_bank=bank)

        trajectories = [
            {"task_id": "t1", "outcome": "success", "task_type": "code"},
            {"task_id": "t2", "outcome": "failure", "task_type": "debug"},
            {"task_id": "t3", "outcome": "success", "escalations": ["a->b"], "task_type": "arch"},
        ]

        report = asyncio.run(pipeline.run(trajectories))

        assert report.skills_stored == 3
        assert bank.count() == 3

    def test_pipeline_teacher_error_handled(self, bank):
        from orchestration.repl_memory.distillation.pipeline import DistillationPipeline

        class FailingTeacher:
            model_id = "failing"
            async def distill(self, prompt, max_tokens=4096):
                raise ConnectionError("API down")

        pipeline = DistillationPipeline(teacher=FailingTeacher(), skill_bank=bank)
        trajectories = [
            {"task_id": "t1", "outcome": "success", "task_type": "code"},
        ]

        report = asyncio.run(pipeline.run(trajectories))

        assert len(report.errors) >= 1
        assert "API down" in report.errors[0]
        assert bank.count() == 0

    def test_pipeline_no_valid_skills_from_response(self, bank):
        from orchestration.repl_memory.distillation.teachers import MockTeacher
        from orchestration.repl_memory.distillation.pipeline import DistillationPipeline

        teacher = MockTeacher(responses=["No patterns found."])
        pipeline = DistillationPipeline(teacher=teacher, skill_bank=bank)

        trajectories = [
            {"task_id": "t1", "outcome": "success", "task_type": "misc"},
        ]

        report = asyncio.run(pipeline.run(trajectories))

        assert report.skills_proposed == 0
        assert report.skills_stored == 0

    def test_pipeline_rejects_empty_principle(self, bank):
        from orchestration.repl_memory.distillation.teachers import MockTeacher
        from orchestration.repl_memory.distillation.pipeline import DistillationPipeline

        teacher_response = self._make_mock_response([
            {"title": "", "principle": "", "when_to_apply": "always",
             "task_types": ["*"], "source_outcome": "success"}
        ])
        teacher = MockTeacher(responses=[teacher_response])
        pipeline = DistillationPipeline(teacher=teacher, skill_bank=bank)

        trajectories = [
            {"task_id": "t1", "outcome": "success", "task_type": "code"},
        ]

        report = asyncio.run(pipeline.run(trajectories))

        assert report.skills_rejected == 1
        assert report.skills_stored == 0

    def test_pipeline_batching(self, bank):
        from orchestration.repl_memory.distillation.teachers import MockTeacher
        from orchestration.repl_memory.distillation.pipeline import DistillationPipeline

        resp = self._make_mock_response([
            {"title": "Batch Skill", "principle": "P", "when_to_apply": "W",
             "task_types": ["*"], "source_outcome": "success"}
        ])
        teacher = MockTeacher(responses=[resp, resp, resp])
        pipeline = DistillationPipeline(
            teacher=teacher, skill_bank=bank, batch_size=5
        )

        # 12 success trajectories → 3 batches of 5, 5, 2
        trajectories = [
            {"task_id": f"t{i}", "outcome": "success", "task_type": "code"}
            for i in range(12)
        ]

        report = asyncio.run(pipeline.run(trajectories))

        assert report.batches_processed == 3
        assert len(teacher.prompts) == 3

    def test_pipeline_report_to_dict(self, bank):
        from orchestration.repl_memory.distillation.pipeline import DistillationReport

        report = DistillationReport(
            total_trajectories=100,
            skills_stored=10,
            duration_seconds=5.123456,
        )
        d = report.to_dict()
        assert d["total_trajectories"] == 100
        assert d["skills_stored"] == 10
        assert d["duration_seconds"] == 5.12


# ── CodexTeacher Config Tests ───────────────────────────────────────────


class TestCodexTeacherConfig:
    """Tests for CodexTeacher configuration (no live calls)."""

    def test_default_binary_path(self):
        from orchestration.repl_memory.distillation.teachers import CodexTeacher

        teacher = CodexTeacher()
        assert teacher.model_id == "gpt-5.3-codex"
        assert teacher._binary.endswith("codex")

    def test_custom_model(self):
        from orchestration.repl_memory.distillation.teachers import CodexTeacher

        teacher = CodexTeacher(model="o3")
        assert teacher.model_id == "o3"

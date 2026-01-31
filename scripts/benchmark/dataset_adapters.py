#!/usr/bin/env python3
"""On-the-fly benchmark question sampling from HuggingFace datasets.

Provides adapters for all debug suite categories, loading real benchmark
questions directly from HuggingFace cached datasets. Each adapter knows
how to extract (question, expected_answer, scoring_method, scoring_config)
from its source dataset's schema.

Supported suites and their data sources:
  - general:              MMLU (cais/mmlu, 14,042 questions)
  - math:                 GSM8K (gsm8k, 1,319) + MATH-500 (HuggingFaceH4/MATH-500, 500)
  - coder:                HumanEval (openai_humaneval, 164) + MBPP (mbpp, 500)
  - thinking:             ARC-Challenge (allenai/ai2_arc, 1,172) + HellaSwag (Rowan/hellaswag, 10,042)
  - instruction_precision: IFEval (google/IFEval, 541)
  - vl:                   OCRBench + ChartQA (via extract_vl_debug_suite.py, 3,500)
  - agentic:              No public dataset (stays YAML-based)
  - long_context:         Synthetic (stays YAML-based)

Usage:
    from dataset_adapters import get_adapter, ADAPTER_SUITES

    adapter = get_adapter("math")
    questions = adapter.sample(n=10, seed=42)
    # Returns list of prompt dicts compatible with compare_orchestrator_direct.py
"""

import random
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Suites that have dataset adapters (vs YAML-only)
ADAPTER_SUITES = {"general", "math", "coder", "thinking", "instruction_precision", "vl"}

# Suites that stay YAML-based (no public dataset or intentionally synthetic)
YAML_ONLY_SUITES = {"agentic", "long_context"}


def get_adapter(suite: str) -> Optional["BaseAdapter"]:
    """Get the dataset adapter for a suite, or None if YAML-only."""
    adapters = {
        "general": MMLUAdapter,
        "math": MathAdapter,
        "coder": CoderAdapter,
        "thinking": ThinkingAdapter,
        "instruction_precision": IFEvalAdapter,
        "vl": VLAdapter,
    }
    cls = adapters.get(suite)
    if cls is None:
        return None
    return cls()


class BaseAdapter:
    """Base class for dataset adapters."""

    suite_name: str = ""
    _dataset = None

    # Adapters with real difficulty data should set this True
    has_real_tiers: bool = False

    def _ensure_loaded(self):
        raise NotImplementedError

    @property
    def total_available(self) -> int:
        self._ensure_loaded()
        return len(self._dataset) if self._dataset is not None else 0

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        raise NotImplementedError

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        """Sample n questions. If stratify=True AND adapter has real tiers,
        draw equal counts per tier for balanced difficulty distribution."""
        self._ensure_loaded()
        if not self._dataset:
            return []
        if stratify and self.has_real_tiers:
            return self._stratified_sample(n, seed)
        rng = random.Random(seed)
        indices = rng.sample(range(len(self._dataset)), min(n, len(self._dataset)))
        return [self._row_to_prompt(i, self._dataset[i]) for i in indices]

    def _stratified_sample(self, n: int, seed: int) -> list[dict]:
        """Draw equal questions per tier. Requires _get_tier_for_index()."""
        rng = random.Random(seed)
        # Bucket indices by tier
        tier_buckets: dict[int, list[int]] = {}
        for i in range(len(self._dataset)):
            t = self._get_tier_for_index(i)
            tier_buckets.setdefault(t, []).append(i)

        tiers = sorted(tier_buckets.keys())
        if not tiers:
            return []

        # Equal share per tier, remainder distributed round-robin
        per_tier = n // len(tiers)
        remainder = n % len(tiers)

        results = []
        for i, t in enumerate(tiers):
            bucket = tier_buckets[t]
            count = per_tier + (1 if i < remainder else 0)
            count = min(count, len(bucket))
            indices = rng.sample(bucket, count)
            results.extend(self._row_to_prompt(idx, self._dataset[idx]) for idx in indices)

        rng.shuffle(results)
        return results

    def _get_tier_for_index(self, idx: int) -> int:
        """Return tier for a given dataset index. Override in adapters with real tiers."""
        return 1


# ── MMLU (General Knowledge) ─────────────────────────────────────────────


class MMLUAdapter(BaseAdapter):
    """MMLU: 14,042 multiple-choice questions across 57 subjects."""

    suite_name = "general"
    has_real_tiers = True  # Subject-based difficulty mapping
    CHOICE_LABELS = ["A", "B", "C", "D"]

    HARD_SUBJECTS = {
        "abstract_algebra", "college_mathematics", "formal_logic",
        "college_physics", "electrical_engineering", "machine_learning",
        "conceptual_physics", "college_chemistry", "anatomy",
    }
    EASY_SUBJECTS = {
        "high_school_geography", "high_school_us_history",
        "miscellaneous", "us_foreign_policy",
    }

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset("cais/mmlu", "all", split="test")
        except Exception as e:
            print(f"  [adapter] MMLU load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row["question"]
        choices = row["choices"]
        answer_idx = row["answer"]
        subject = row.get("subject", "general")

        # Build multiple-choice prompt
        prompt_lines = [question, ""]
        for i, choice in enumerate(choices):
            prompt_lines.append(f"{self.CHOICE_LABELS[i]}) {choice}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only (A, B, C, or D).")

        expected = self.CHOICE_LABELS[answer_idx]

        # Tier based on subject difficulty
        if subject in self.HARD_SUBJECTS:
            tier = 3
        elif subject in self.EASY_SUBJECTS:
            tier = 1
        else:
            tier = 2

        return {
            "id": f"mmlu_{subject}_{idx:05d}",
            "suite": "general",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }

    def _get_tier_for_index(self, idx: int) -> int:
        subject = self._dataset[idx].get("subject", "general")
        if subject in self.HARD_SUBJECTS:
            return 3
        elif subject in self.EASY_SUBJECTS:
            return 1
        return 2


# ── GSM8K + MATH-500 (Math) ──────────────────────────────────────────────


class MathAdapter(BaseAdapter):
    """GSM8K (1,319) + MATH-500 (500) = 1,819 math problems."""

    suite_name = "math"
    has_real_tiers = True  # GSM8K=T1, MATH-500 level 1-3=T2, level 4-5=T3
    _gsm8k = None
    _math500 = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._gsm8k = hf.load_dataset("gsm8k", "main", split="test")
            try:
                self._math500 = hf.load_dataset("HuggingFaceH4/MATH-500", split="test")
            except Exception:
                self._math500 = []
            # Combine into unified list
            self._dataset = list(range(len(self._gsm8k) + len(self._math500)))
        except Exception as e:
            print(f"  [adapter] Math datasets load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        # idx is from unified list; row is ignored, we index directly
        gsm8k_len = len(self._gsm8k) if self._gsm8k else 0

        if idx < gsm8k_len:
            return self._gsm8k_prompt(idx, self._gsm8k[idx])
        else:
            math_idx = idx - gsm8k_len
            return self._math500_prompt(math_idx, self._math500[math_idx])

    def _get_tier_for_index(self, idx: int) -> int:
        gsm8k_len = len(self._gsm8k) if self._gsm8k else 0
        if idx < gsm8k_len:
            return 1  # GSM8K = grade-school
        math_idx = idx - gsm8k_len
        if self._math500 and math_idx < len(self._math500):
            level = self._math500[math_idx].get("level", 3)
            return 2 if level <= 3 else 3
        return 1

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if not self._dataset:
            return []
        if stratify:
            return self._stratified_sample(n, seed)
        rng = random.Random(seed)
        # Split: ~60% GSM8K, ~40% MATH-500
        gsm8k_len = len(self._gsm8k) if self._gsm8k else 0
        math_len = len(self._math500) if self._math500 else 0

        n_gsm = min(int(n * 0.6), gsm8k_len)
        n_math = min(n - n_gsm, math_len)
        if n_math < n - n_gsm:
            n_gsm = min(n - n_math, gsm8k_len)

        results = []
        if n_gsm > 0:
            gsm_indices = rng.sample(range(gsm8k_len), n_gsm)
            results.extend(self._gsm8k_prompt(i, self._gsm8k[i]) for i in gsm_indices)
        if n_math > 0:
            math_indices = rng.sample(range(math_len), n_math)
            results.extend(self._math500_prompt(i, self._math500[i]) for i in math_indices)

        rng.shuffle(results)
        return results

    @staticmethod
    def _extract_gsm8k_answer(answer_text: str) -> str:
        """Extract numeric answer from GSM8K solution (after ####)."""
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return match.group(1).strip().replace(",", "")
        return answer_text.strip()

    def _gsm8k_prompt(self, idx: int, row: dict) -> dict:
        question = row["question"]
        answer_text = row["answer"]
        expected = self._extract_gsm8k_answer(answer_text)

        return {
            "id": f"gsm8k_{idx:05d}",
            "suite": "math",
            "prompt": question + "\n\nSolve step by step. Put your final numeric answer after ####.",
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": 1,  # GSM8K is grade-school level
            "scoring_method": "exact_match",
            "scoring_config": {"extract_pattern": r"####\s*([^\s]+)"},
        }

    def _math500_prompt(self, idx: int, row: dict) -> dict:
        problem = row["problem"]
        answer = row.get("answer", "")
        level = row.get("level", 3)
        subject = row.get("subject", "")

        # Map MATH difficulty level to tier
        tier = 2 if level <= 3 else 3

        return {
            "id": f"math500_{subject}_{idx:05d}",
            "suite": "math",
            "prompt": problem + "\n\nPut your final answer in \\boxed{}.",
            "context": "",
            "expected": answer,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "substring",
            "scoring_config": {"case_sensitive": False},
        }


# ── HumanEval + MBPP (Coder) ─────────────────────────────────────────────


class CoderAdapter(BaseAdapter):
    """HumanEval (164) + MBPP (500) = 664 coding problems."""

    suite_name = "coder"
    _humaneval = None
    _mbpp = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._humaneval = hf.load_dataset("openai_humaneval", split="test")
            self._mbpp = hf.load_dataset("mbpp", split="test")
            self._dataset = list(range(len(self._humaneval) + len(self._mbpp)))
        except Exception as e:
            print(f"  [adapter] Coder datasets load failed: {e}")
            self._dataset = []

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if not self._dataset:
            return []
        rng = random.Random(seed)
        he_len = len(self._humaneval) if self._humaneval else 0
        mbpp_len = len(self._mbpp) if self._mbpp else 0

        n_he = min(int(n * 0.4), he_len)
        n_mbpp = min(n - n_he, mbpp_len)
        if n_mbpp < n - n_he:
            n_he = min(n - n_mbpp, he_len)

        results = []
        if n_he > 0:
            he_indices = rng.sample(range(he_len), n_he)
            results.extend(self._humaneval_prompt(i) for i in he_indices)
        if n_mbpp > 0:
            mbpp_indices = rng.sample(range(mbpp_len), n_mbpp)
            results.extend(self._mbpp_prompt(i) for i in mbpp_indices)

        rng.shuffle(results)
        return results

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        he_len = len(self._humaneval) if self._humaneval else 0
        if idx < he_len:
            return self._humaneval_prompt(idx)
        return self._mbpp_prompt(idx - he_len)

    def _humaneval_prompt(self, idx: int) -> dict:
        row = self._humaneval[idx]
        prompt_text = row["prompt"]
        canonical = row["canonical_solution"]
        test_code = row["test"]
        entry_point = row["entry_point"]
        task_id = row["task_id"]

        # Build a prompt that asks to complete the function
        full_prompt = (
            f"Complete the following Python function:\n\n"
            f"```python\n{prompt_text}```\n\n"
            f"Write only the function body (the part after the signature)."
        )

        return {
            "id": f"humaneval_{task_id.replace('/', '_')}",
            "suite": "coder",
            "prompt": full_prompt,
            "context": "",
            "expected": entry_point,
            "scoring": [],
            "image_path": "",
            "tier": 2,
            "scoring_method": "substring",
            "scoring_config": {"case_sensitive": True, "substring": entry_point},
        }

    def _mbpp_prompt(self, idx: int) -> dict:
        row = self._mbpp[idx]
        task_id = row["task_id"]
        text = row["text"]
        test_list = row.get("test_list", [])

        # Include test cases as hints
        test_hint = ""
        if test_list:
            test_hint = "\n\nTest cases:\n" + "\n".join(f"  {t}" for t in test_list[:3])

        full_prompt = (
            f"{text}{test_hint}\n\n"
            f"Write a Python function to solve this."
        )

        # Extract expected function name from test cases
        func_name = ""
        if test_list:
            match = re.search(r"assert\s+(\w+)\(", test_list[0])
            if match:
                func_name = match.group(1)

        return {
            "id": f"mbpp_{task_id:04d}",
            "suite": "coder",
            "prompt": full_prompt,
            "context": "",
            "expected": func_name or "def",
            "scoring": [],
            "image_path": "",
            "tier": 1,
            "scoring_method": "substring",
            "scoring_config": {"case_sensitive": True, "substring": func_name or "def"},
        }


# ── ARC-Challenge + HellaSwag (Thinking) ──────────────────────────────────


class ThinkingAdapter(BaseAdapter):
    """ARC-Challenge (1,172) + HellaSwag (10,042) = 11,214 reasoning questions."""

    suite_name = "thinking"
    _arc = None
    _hellaswag = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._arc = hf.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            self._hellaswag = hf.load_dataset("Rowan/hellaswag", split="validation")
            self._dataset = list(range(len(self._arc) + len(self._hellaswag)))
        except Exception as e:
            print(f"  [adapter] Thinking datasets load failed: {e}")
            self._dataset = []

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if not self._dataset:
            return []
        rng = random.Random(seed)
        arc_len = len(self._arc) if self._arc else 0
        hs_len = len(self._hellaswag) if self._hellaswag else 0

        n_arc = min(int(n * 0.5), arc_len)
        n_hs = min(n - n_arc, hs_len)
        if n_hs < n - n_arc:
            n_arc = min(n - n_hs, arc_len)

        results = []
        if n_arc > 0:
            arc_indices = rng.sample(range(arc_len), n_arc)
            results.extend(self._arc_prompt(i) for i in arc_indices)
        if n_hs > 0:
            hs_indices = rng.sample(range(hs_len), n_hs)
            results.extend(self._hellaswag_prompt(i) for i in hs_indices)

        rng.shuffle(results)
        return results

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        arc_len = len(self._arc) if self._arc else 0
        if idx < arc_len:
            return self._arc_prompt(idx)
        return self._hellaswag_prompt(idx - arc_len)

    CHOICE_LABELS = ["A", "B", "C", "D", "E"]

    def _arc_prompt(self, idx: int) -> dict:
        row = self._arc[idx]
        question = row["question"]
        choices_data = row["choices"]
        answer_key = row["answerKey"]
        qid = row["id"]

        # ARC choices format: {"text": [...], "label": [...]}
        labels = choices_data["label"]
        texts = choices_data["text"]

        prompt_lines = [question, ""]
        for label, text in zip(labels, texts):
            prompt_lines.append(f"{label}) {text}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only.")

        return {
            "id": f"arc_{qid}",
            "suite": "thinking",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": answer_key,
            "scoring": [],
            "image_path": "",
            "tier": 2,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }

    def _hellaswag_prompt(self, idx: int) -> dict:
        row = self._hellaswag[idx]
        context = row["ctx"]
        endings = row["endings"]
        label = row["label"]
        ind = row["ind"]

        prompt_lines = [
            "Choose the most plausible continuation:",
            "",
            f"Context: {context}",
            "",
        ]
        for i, ending in enumerate(endings):
            prompt_lines.append(f"{self.CHOICE_LABELS[i]}) {ending}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only (A, B, C, or D).")

        expected = self.CHOICE_LABELS[int(label)] if isinstance(label, (int, str)) else "A"

        return {
            "id": f"hellaswag_{ind:05d}",
            "suite": "thinking",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": 1,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }


# ── IFEval (Instruction Precision) ───────────────────────────────────────


class IFEvalAdapter(BaseAdapter):
    """IFEval: 541 instruction-following prompts with verifiable constraints."""

    suite_name = "instruction_precision"
    has_real_tiers = True  # Tier from constraint count: 1→T1, 2-3→T2, 4+→T3

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset("google/IFEval", split="train")
        except Exception as e:
            print(f"  [adapter] IFEval load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        prompt = row["prompt"]
        key = row["key"]
        instruction_ids = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [])

        # IFEval doesn't have simple expected answers — it has constraint verifiers.
        # We extract the first instruction as the primary constraint to check.
        primary_constraint = instruction_ids[0] if instruction_ids else "unknown"

        # Build scoring config from IFEval's constraint types
        scoring_method, scoring_config = self._constraint_to_scoring(
            primary_constraint, kwargs_list[0] if kwargs_list else {}
        )

        # Determine tier from constraint complexity
        n_constraints = len(instruction_ids)
        tier = 1 if n_constraints <= 1 else (2 if n_constraints <= 3 else 3)

        return {
            "id": f"ifeval_{key}",
            "suite": "instruction_precision",
            "prompt": prompt,
            "context": "",
            "expected": "",  # IFEval uses programmatic verification
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": scoring_method,
            "scoring_config": scoring_config,
            "ifeval_instructions": instruction_ids,
            "ifeval_kwargs": kwargs_list,
        }

    @staticmethod
    def _constraint_to_scoring(constraint_id: str, kwargs: dict) -> tuple[str, dict]:
        """Map IFEval constraint type to our scoring system."""
        # IFEval constraints: https://github.com/google-research/google-research/tree/master/instruction_following_eval
        if "no_comma" in constraint_id:
            return "programmatic", {"verifier": "no_comma"}
        elif "number_highlighted_sections" in constraint_id:
            n = kwargs.get("num_highlights", 1)
            return "programmatic", {"verifier": "highlighted_sections", "count": n}
        elif "number_paragraphs" in constraint_id:
            n = kwargs.get("num_paragraphs", 1)
            return "programmatic", {"verifier": "paragraph_count", "count": n}
        elif "number_words" in constraint_id or "length" in constraint_id:
            n = kwargs.get("num_words")
            rel = kwargs.get("relation", "at_least")
            return "programmatic", {"verifier": "word_count", "count": n, "relation": rel}
        elif "number_sentences" in constraint_id:
            n = kwargs.get("num_sentences")
            rel = kwargs.get("relation", "at_least")
            return "programmatic", {"verifier": "sentence_count", "count": n, "relation": rel}
        elif "postscript" in constraint_id:
            return "substring", {"case_sensitive": False, "substring": "P.S."}
        elif "title" in constraint_id:
            return "programmatic", {"verifier": "has_title"}
        elif "json_format" in constraint_id or "json" in constraint_id:
            return "programmatic", {"verifier": "json_valid"}
        elif "number_placeholders" in constraint_id:
            n = kwargs.get("num_placeholders", 1)
            return "programmatic", {"verifier": "placeholder_count", "count": n}
        elif "bullet_list" in constraint_id or "number_bullet" in constraint_id:
            n = kwargs.get("num_bullets")
            return "programmatic", {"verifier": "bullet_count", "count": n}
        elif "keywords" in constraint_id:
            kw = kwargs.get("keywords", [])
            return "programmatic", {"verifier": "contains_keywords", "keywords": kw}
        elif "forbidden" in constraint_id:
            fw = kwargs.get("forbidden_words", [])
            return "programmatic", {"verifier": "no_forbidden_words", "forbidden": fw}
        elif "language" in constraint_id:
            lang = kwargs.get("language", "en")
            return "programmatic", {"verifier": "language", "language": lang}
        else:
            # Generic fallback — just check response is non-empty
            return "programmatic", {"verifier": "non_empty", "constraint": constraint_id}

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        n_constraints = len(row.get("instruction_id_list", []))
        return 1 if n_constraints <= 1 else (2 if n_constraints <= 3 else 3)


# ── VL (Vision-Language) ──────────────────────────────────────────────────


class VLAdapter(BaseAdapter):
    """VL: delegates to extract_vl_debug_suite.VLDatasetAdapter (3,500 questions)."""

    suite_name = "vl"
    _vl_adapter = None

    def _ensure_loaded(self):
        if self._vl_adapter is not None:
            return
        try:
            from extract_vl_debug_suite import VLDatasetAdapter
            self._vl_adapter = VLDatasetAdapter()
            self._dataset = list(range(self._vl_adapter.total_available))
        except ImportError:
            print("  [adapter] VL adapter not available (extract_vl_debug_suite.py)")
            self._dataset = []

    @property
    def total_available(self) -> int:
        self._ensure_loaded()
        return self._vl_adapter.total_available if self._vl_adapter else 0

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if self._vl_adapter:
            return self._vl_adapter.sample(n=n, seed=seed, extract_images=True)
        return []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        return {}  # Not used — sample() delegates directly

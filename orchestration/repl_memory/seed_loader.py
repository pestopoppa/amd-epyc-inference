#!/usr/bin/env python3
"""
Seed the episodic memory with canonical REPL tool usage examples.

Usage:
    python orchestration/repl_memory/seed_loader.py [--force]

Options:
    --force     Clear existing memories and reload all seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.repl_memory.episodic_store import EpisodicStore
from orchestration.repl_memory.embedder import TaskEmbedder

SEED_FILE = Path(__file__).parent / "seed_examples.json"


def load_seeds() -> list[dict]:
    """Load seed examples from JSON file."""
    with open(SEED_FILE) as f:
        return json.load(f)


def _get_routing_seeds() -> list[dict]:
    """Return mode-annotated routing seed examples.

    These teach MemRL which execution mode (direct/react/repl) works
    best for which task types, bootstrapping the mode selection before
    enough real task outcomes accumulate.

    Returns:
        List of routing seed dictionaries.
    """
    return [
        # Direct mode seeds — instruction following, reasoning, formatting
        {"task": "Solve this logic puzzle about truth-tellers and liars",
         "task_type": "reasoning", "action": "frontdoor:direct", "mode": "direct", "outcome": "success"},
        {"task": "Reformat this text as a numbered list with exactly 5 items",
         "task_type": "formatting", "action": "frontdoor:direct", "mode": "direct", "outcome": "success"},
        {"task": "Write a haiku about autumn",
         "task_type": "creative", "action": "frontdoor:direct", "mode": "direct", "outcome": "success"},
        {"task": "Explain the difference between TCP and UDP",
         "task_type": "knowledge", "action": "frontdoor:direct", "mode": "direct", "outcome": "success"},
        {"task": "Prove that the square root of 2 is irrational",
         "task_type": "math", "action": "frontdoor:direct", "mode": "direct", "outcome": "success"},
        {"task": "Generate JSON output with user name and age fields",
         "task_type": "formatting", "action": "frontdoor:direct", "mode": "direct", "outcome": "success"},
        # React mode seeds — tool-needing queries
        {"task": "Search for recent papers on transformer architectures",
         "task_type": "research", "action": "frontdoor:react", "mode": "react", "outcome": "success"},
        {"task": "What is today's date?",
         "task_type": "factual", "action": "frontdoor:react", "mode": "react", "outcome": "success"},
        {"task": "Calculate the compound interest on $10000 at 5% for 10 years",
         "task_type": "math", "action": "frontdoor:react", "mode": "react", "outcome": "success"},
        {"task": "Look up the Wikipedia article about quantum entanglement",
         "task_type": "research", "action": "frontdoor:react", "mode": "react", "outcome": "success"},
        {"task": "Search arXiv for papers about reinforcement learning from human feedback",
         "task_type": "research", "action": "frontdoor:react", "mode": "react", "outcome": "success"},
        # REPL mode seeds — file exploration, large context, code execution
        {"task": "Read the configuration file and summarize its settings",
         "task_type": "file_exploration", "action": "frontdoor:repl", "mode": "repl", "outcome": "success"},
        {"task": "List all Python files in the source directory",
         "task_type": "file_exploration", "action": "frontdoor:repl", "mode": "repl", "outcome": "success"},
        {"task": "Summarize this 50-page document about climate change",
         "task_type": "ingest", "action": "frontdoor:repl", "mode": "repl", "outcome": "success"},
        {"task": "Find all functions that handle error cases in the codebase",
         "task_type": "code_exploration", "action": "frontdoor:repl", "mode": "repl", "outcome": "success"},
        {"task": "Execute the benchmark script and report the results",
         "task_type": "execution", "action": "frontdoor:repl", "mode": "repl", "outcome": "success"},
        {"task": "Grep for all TODO comments in the project",
         "task_type": "code_exploration", "action": "frontdoor:repl", "mode": "repl", "outcome": "success"},
    ]


def seed_memory(force: bool = False) -> dict:
    """
    Load seed examples into episodic memory.

    Args:
        force: If True, clear existing memories first

    Returns:
        Stats dict with counts
    """
    store = EpisodicStore()
    embedder = TaskEmbedder()

    # Check current state
    current_stats = store.get_stats()
    current_count = current_stats.get("total_memories", 0)

    if current_count > 0 and not force:
        print(f"Memory already has {current_count} entries.")
        print("Use --force to clear and reload all seeds.")
        return {"skipped": True, "existing": current_count}

    if force and current_count > 0:
        print(f"Clearing {current_count} existing memories...")
        # Clear by removing the SQLite database and FAISS files
        # FAISS backend uses storage_dir/episodic.db and storage_dir/embeddings.faiss
        sqlite_path = store.sqlite_path
        faiss_path = store.storage_dir / "embeddings.faiss"
        id_map_path = store.storage_dir / "id_map.npy"

        store.close()  # Close before deleting

        sqlite_path.unlink(missing_ok=True)
        faiss_path.unlink(missing_ok=True)
        id_map_path.unlink(missing_ok=True)

        store = EpisodicStore()  # Reinitialize

    # Load and embed seed examples
    seeds = load_seeds()
    print(f"Loading {len(seeds)} seed examples...")

    stats = {
        "loaded": 0,
        "failed": 0,
        "by_category": {},
    }

    for i, seed in enumerate(seeds):
        task = seed["task"]
        code = seed["code"]
        category = seed.get("category", "unknown")
        tools_used = seed.get("tools_used", [])

        try:
            # Generate embedding for the task description
            embedding = embedder.embed_text(task)

            # Store in episodic memory
            context = {
                "task_description": task,
                "category": category,
                "tools_used": tools_used,
                "is_seed": True,
            }

            store.store(
                embedding=embedding,
                action=code,
                action_type="exploration",  # REPL code is exploration type
                context=context,
                outcome="success",  # Seed examples are all successful
                initial_q=0.9,  # High Q-value for known-good examples
            )

            stats["loaded"] += 1
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(seeds)} examples...")

        except Exception as e:
            print(f"  Failed to load '{task[:50]}...': {e}")
            stats["failed"] += 1

    # Load mode-annotated routing seeds
    routing_seeds = _get_routing_seeds()
    for rseed in routing_seeds:
        try:
            embedding = embedder.embed_text(rseed["task"])
            context = {
                "task_description": rseed["task"],
                "task_type": rseed["task_type"],
                "category": "routing",
                "mode": rseed["mode"],
                "is_seed": True,
            }
            store.store(
                embedding=embedding,
                action=rseed["action"],
                action_type="routing",
                context=context,
                outcome=rseed.get("outcome", "success"),
                initial_q=0.85,
            )
            stats["loaded"] += 1
            stats["by_category"]["routing"] = stats["by_category"].get("routing", 0) + 1
        except Exception as e:
            print(f"  Failed to load routing seed '{rseed['task'][:50]}...': {e}")
            stats["failed"] += 1

    # Flush FAISS index to disk before reporting stats
    store.flush()
    store._embedding_store.save()

    print(f"\nSeeding complete!")
    print(f"  Loaded: {stats['loaded']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  By category: {stats['by_category']}")

    # Show final stats
    final_stats = store.get_stats()
    print(f"\nMemory stats:")
    print(f"  Total memories: {final_stats['total_memories']}")
    print(f"  FAISS embeddings: {store._embedding_store.count}")
    print(f"  Average Q-value: {final_stats['overall_avg_q']:.2f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Seed episodic memory with REPL examples")
    parser.add_argument("--force", action="store_true", help="Clear existing memories first")
    args = parser.parse_args()

    seed_memory(force=args.force)


if __name__ == "__main__":
    main()

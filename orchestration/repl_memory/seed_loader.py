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
        # Clear by recreating the store files
        store.db_path.unlink(missing_ok=True)
        store.embeddings_path.unlink(missing_ok=True)
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

    print(f"\nSeeding complete!")
    print(f"  Loaded: {stats['loaded']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  By category: {stats['by_category']}")

    # Show final stats
    final_stats = store.get_stats()
    print(f"\nMemory stats:")
    print(f"  Total memories: {final_stats['total_memories']}")
    print(f"  Average Q-value: {final_stats['overall_avg_q']:.2f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Seed episodic memory with REPL examples")
    parser.add_argument("--force", action="store_true", help="Clear existing memories first")
    args = parser.parse_args()

    seed_memory(force=args.force)


if __name__ == "__main__":
    main()

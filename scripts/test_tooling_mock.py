#!/usr/bin/env python3
"""Mock test of orchestrator tooling infrastructure.

Demonstrates:
1. Tool registry setup with role permissions
2. Script registry loading
3. REPL integration with TOOL/SCRIPT functions
4. Permission checking by role tier
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tool_registry import ToolRegistry, ToolPermissions, ToolCategory
from src.script_registry import ScriptRegistry
from src.repl_environment import REPLEnvironment, REPLConfig
from src.tools import register_all_tools


def setup_registries():
    """Set up tool and script registries with permissions."""
    print("=" * 60)
    print("SETTING UP REGISTRIES")
    print("=" * 60)

    # Create tool registry
    tool_registry = ToolRegistry()

    # Register built-in tools
    tool_count = register_all_tools(tool_registry)
    print(f"✓ Registered {tool_count} tools")

    # Set up role permissions (from model_registry.yaml)
    tool_registry.set_role_permissions(
        "frontdoor",
        ToolPermissions(
            web_access=True,
            allowed_categories=[ToolCategory.WEB, ToolCategory.FILE, ToolCategory.DATA],
        ),
    )
    tool_registry.set_role_permissions(
        "coder_primary",
        ToolPermissions(
            web_access=True,
            allowed_categories=[
                ToolCategory.WEB,
                ToolCategory.FILE,
                ToolCategory.CODE,
                ToolCategory.DATA,
                ToolCategory.SYSTEM,
            ],
            allowed_tools=["run_tests", "lint_python", "fetch_docs"],
        ),
    )
    tool_registry.set_role_permissions(
        "worker_general",
        ToolPermissions(
            web_access=False,
            allowed_categories=[ToolCategory.FILE, ToolCategory.DATA],
            forbidden_tools=["write_file"],
        ),
    )
    print("✓ Configured role permissions")

    # Create script registry
    script_registry = ScriptRegistry()

    # Load prepared scripts
    script_dir = Path("/mnt/raid0/llm/claude/orchestration/script_registry")
    if script_dir.exists():
        script_count = script_registry.load_from_directory(script_dir)
        print(f"✓ Loaded {script_count} prepared scripts")
    else:
        print("⚠ Script directory not found, skipping script loading")

    return tool_registry, script_registry


def test_tool_permissions(tool_registry: ToolRegistry):
    """Test tool permission checking for different roles."""
    print("\n" + "=" * 60)
    print("TESTING TOOL PERMISSIONS")
    print("=" * 60)

    test_cases = [
        ("frontdoor", "fetch_docs", True),
        ("frontdoor", "read_file", True),
        ("coder_primary", "run_tests", True),
        ("coder_primary", "lint_python", True),
        ("worker_general", "fetch_docs", False),  # No web access
        ("worker_general", "read_file", True),
        ("worker_general", "json_parse", True),
    ]

    for role, tool, expected in test_cases:
        can_use = tool_registry.can_use_tool(role, tool)
        status = "✓" if can_use == expected else "✗"
        result = "allowed" if can_use else "denied"
        print(f"  {status} {role} -> {tool}: {result}")


def test_script_discovery(script_registry: ScriptRegistry):
    """Test script discovery via fuzzy search."""
    print("\n" + "=" * 60)
    print("TESTING SCRIPT DISCOVERY")
    print("=" * 60)

    queries = [
        "fetch python documentation",
        "run pytest tests",
        "parse json data",
        "arxiv papers",
    ]

    for query in queries:
        matches = script_registry.find_scripts(query, limit=2)
        print(f"\n  Query: '{query}'")
        for match in matches:
            print(f"    → {match.script.id} (score: {match.score:.2f}, matched: {match.matched_on})")


def test_repl_integration(tool_registry: ToolRegistry, script_registry: ScriptRegistry):
    """Test REPL environment with TOOL/SCRIPT functions."""
    print("\n" + "=" * 60)
    print("TESTING REPL INTEGRATION")
    print("=" * 60)

    # Test as frontdoor role (Tier A - full access)
    print("\n--- Testing as 'frontdoor' role (Tier A) ---")
    repl = REPLEnvironment(
        context="Test context for frontdoor",
        tool_registry=tool_registry,
        script_registry=script_registry,
        role="frontdoor",
    )

    # Test list_tools
    result = repl.execute("tools = list_tools(); print(f'Available tools: {len(tools)}')")
    print(f"  list_tools(): {result.output.strip()}")

    # Test find_scripts
    result = repl.execute("matches = find_scripts('fetch docs'); print(f'Found {len(matches)} matches')")
    print(f"  find_scripts(): {result.output.strip()}")

    # Test TOOL invocation (read_file on a real file)
    result = repl.execute("""
result = TOOL('read_file', file_path='/mnt/raid0/llm/claude/README.md', limit=5)
if result['success']:
    print(f"Read {result['lines_returned']} lines from README.md")
else:
    print(f"Error: {result.get('error')}")
""")
    print(f"  TOOL('read_file'): {result.output.strip()}")

    # Test as worker_general role (Tier C - restricted)
    print("\n--- Testing as 'worker_general' role (Tier C) ---")
    worker_repl = REPLEnvironment(
        context="Test context for worker",
        tool_registry=tool_registry,
        script_registry=script_registry,
        role="worker_general",
    )

    # Workers should have fewer tools
    result = worker_repl.execute("tools = list_tools(); print(f'Available tools: {len(tools)}')")
    print(f"  list_tools(): {result.output.strip()}")

    # Test permission denial for web tool
    result = worker_repl.execute("""
try:
    TOOL('fetch_docs', url='https://example.com')
    print("Unexpected: web tool allowed!")
except PermissionError as e:
    print(f"Correctly denied: {e}")
except Exception as e:
    print(f"Error: {e}")
""")
    print(f"  TOOL('fetch_docs'): {result.output.strip()}")


def test_script_invocation(script_registry: ScriptRegistry):
    """Test direct script invocation."""
    print("\n" + "=" * 60)
    print("TESTING SCRIPT INVOCATION")
    print("=" * 60)

    # Test parse_json script (has embedded code)
    test_json = '{"name": "test", "value": 42, "nested": {"key": "val"}}'

    print(f"\n  Input JSON: {test_json}")

    # Parse without extraction
    try:
        result = script_registry.invoke("parse_json", content=test_json)
        print(f"  parse_json(): {result}")
    except Exception as e:
        print(f"  parse_json() error: {e}")

    # Parse with path extraction
    try:
        result = script_registry.invoke("parse_json", content=test_json, extract_path="nested.key")
        print(f"  parse_json(extract='nested.key'): {result}")
    except Exception as e:
        print(f"  parse_json() error: {e}")


def test_tool_invocation(tool_registry: ToolRegistry):
    """Test direct tool invocation."""
    print("\n" + "=" * 60)
    print("TESTING TOOL INVOCATION")
    print("=" * 60)

    # Test list_dir tool
    print("\n  Testing list_dir on /mnt/raid0/llm/claude/src/")
    try:
        result = tool_registry.invoke(
            "list_dir",
            "frontdoor",
            directory="/mnt/raid0/llm/claude/src/",
            limit=10,
        )
        if result["success"]:
            print(f"  ✓ Found {result['returned_count']} entries:")
            for entry in result["entries"][:5]:
                print(f"    - {entry['name']} ({entry['type']})")
        else:
            print(f"  ✗ Error: {result.get('error')}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")

    # Test json_parse tool
    print("\n  Testing json_parse:")
    try:
        result = tool_registry.invoke(
            "json_parse",
            "frontdoor",
            content='{"test": 123}',
        )
        print(f"  ✓ Result: {result}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")


def print_summary(tool_registry: ToolRegistry, script_registry: ScriptRegistry):
    """Print summary of available tools and scripts."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nRegistered Tools:")
    for tool_info in tool_registry.list_tools():
        print(f"  - {tool_info['name']} ({tool_info['category']})")

    print("\nRegistered Scripts:")
    for script_info in script_registry.list_scripts():
        print(f"  - {script_info['id']} ({script_info['category']}) - {script_info.get('token_savings', 'N/A')}")

    print("\nScript Categories:", script_registry.get_categories())
    print("Script Tags:", script_registry.get_tags())


def main():
    """Run all mock tests."""
    print("\n" + "=" * 60)
    print("ORCHESTRATOR TOOLING MOCK TEST")
    print("=" * 60)

    # Setup
    tool_registry, script_registry = setup_registries()

    # Tests
    test_tool_permissions(tool_registry)
    test_script_discovery(script_registry)
    test_script_invocation(script_registry)
    test_tool_invocation(tool_registry)
    test_repl_integration(tool_registry, script_registry)

    # Summary
    print_summary(tool_registry, script_registry)

    print("\n" + "=" * 60)
    print("MOCK TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

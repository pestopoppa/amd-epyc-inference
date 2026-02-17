#!/usr/bin/env python3
"""Unit tests for the REPL procedure tools (_ProcedureToolsMixin)."""

from unittest.mock import Mock, patch, mock_open

from src.repl_environment import REPLEnvironment


class TestRunProcedure:
    """Test _run_procedure() / run_procedure() function."""

    @patch("orchestration.procedure_registry.ProcedureRegistry")
    def test_run_procedure_success(self, mock_registry_class):
        """Test run_procedure() executes successfully."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        # Mock successful execution result
        mock_result = Mock()
        mock_result.success = True
        mock_result.procedure_id = "test_proc"
        mock_result.error = None
        mock_result.elapsed_seconds = 0.5
        mock_result.outputs = {"result": "success"}
        mock_result.step_results = [Mock(success=True), Mock(success=True)]
        mock_registry.execute.return_value = mock_result

        repl = REPLEnvironment(context="test", role="worker_general")
        result = repl.execute("""
output = run_procedure('test_proc', arg1='value1')
data = json.loads(output)
print(data['success'])
print(data['steps_completed'])
""")

        assert result.error is None
        assert "True" in result.output
        assert "2" in result.output

    @patch("orchestration.procedure_registry.ProcedureRegistry")
    def test_run_procedure_failure(self, mock_registry_class):
        """Test run_procedure() handles execution failures."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_result = Mock()
        mock_result.success = False
        mock_result.procedure_id = "test_proc"
        mock_result.error = "Step 2 failed"
        mock_result.elapsed_seconds = 0.3
        mock_result.outputs = {}
        mock_result.step_results = [Mock(success=True), Mock(success=False)]
        mock_registry.execute.return_value = mock_result

        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = run_procedure('test_proc')
data = json.loads(output)
print(data['success'])
print(data['error'])
""")

        assert result.error is None
        assert "False" in result.output
        assert "Step 2 failed" in result.output

    def test_run_procedure_increments_exploration(self):
        """Test run_procedure() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        # Will fail to import but still increment
        repl.execute("run_procedure('test')")

        assert repl._exploration_calls > initial_calls


class TestListProcedures:
    """Test _list_procedures() / list_procedures() function."""

    @patch("orchestration.procedure_registry.ProcedureRegistry")
    def test_list_procedures_all(self, mock_registry_class):
        """Test list_procedures() returns all procedures."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_procedures = [
            {"id": "proc1", "category": "checkpoint", "description": "Create checkpoint"},
            {"id": "proc2", "category": "benchmark", "description": "Run benchmark"},
        ]
        mock_registry.list_procedures.return_value = mock_procedures

        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = list_procedures()
data = json.loads(output)
print(len(data))
""")

        assert result.error is None
        assert "2" in result.output

    @patch("orchestration.procedure_registry.ProcedureRegistry")
    def test_list_procedures_filtered(self, mock_registry_class):
        """Test list_procedures() with category filter."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_procedures = [{"id": "checkpoint_create", "category": "checkpoint"}]
        mock_registry.list_procedures.return_value = mock_procedures

        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = list_procedures(category='checkpoint')
data = json.loads(output)
print(len(data))
""")

        assert result.error is None
        assert "1" in result.output

    def test_list_procedures_increments_exploration(self):
        """Test list_procedures() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("list_procedures()")

        assert repl._exploration_calls > initial_calls


class TestGetProcedureStatus:
    """Test _get_procedure_status() / get_procedure_status() function."""

    def test_get_procedure_status_never_run(self):
        """Test get_procedure_status() for never-run procedure."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = get_procedure_status('nonexistent_proc')
data = json.loads(output)
print(data.get('status'))
""")

        assert result.error is None
        assert "never_run" in result.output

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"status": "completed", "timestamp": "2024-01-01"}',
    )
    def test_get_procedure_status_with_history(self, mock_file, mock_glob, mock_exists):
        """Test get_procedure_status() with execution history."""
        mock_exists.return_value = True
        mock_state_file = Mock()
        mock_state_file.name = "test_proc_20240101.json"
        mock_glob.return_value = [mock_state_file]

        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = get_procedure_status('test_proc')
data = json.loads(output)
print(data.get('status'))
""")

        assert result.error is None
        assert "completed" in result.output

    def test_get_procedure_status_increments_exploration(self):
        """Test get_procedure_status() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("get_procedure_status('test')")

        assert repl._exploration_calls > initial_calls


class TestCheckpointCreate:
    """Test _checkpoint_create() / checkpoint_create() function."""

    @patch("orchestration.procedure_registry.ProcedureRegistry")
    def test_checkpoint_create(self, mock_registry_class):
        """Test checkpoint_create() delegates to run_procedure."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_result = Mock()
        mock_result.success = True
        mock_result.procedure_id = "checkpoint_create"
        mock_result.error = None
        mock_result.elapsed_seconds = 0.2
        mock_result.outputs = {"checkpoint_id": "ckpt_123"}
        mock_result.step_results = [Mock(success=True)]
        mock_registry.execute.return_value = mock_result

        repl = REPLEnvironment(context="test")
        result = repl.execute("output = checkpoint_create('test checkpoint')")

        assert result.error is None


class TestCheckpointRestore:
    """Test _checkpoint_restore() / checkpoint_restore() function."""

    @patch("pathlib.Path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"checkpoint_id": "ckpt_123", "created_at": "2024-01-01T00:00:00"}',
    )
    def test_checkpoint_restore_success(self, mock_file, mock_exists):
        """Test checkpoint_restore() successfully restores."""
        mock_exists.return_value = True

        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = checkpoint_restore('ckpt_123')
data = json.loads(output)
print(data.get('restored'))
print(data.get('checkpoint_id'))
""")

        assert result.error is None
        assert "True" in result.output
        assert "ckpt_123" in result.output

    @patch("pathlib.Path.exists")
    def test_checkpoint_restore_not_found(self, mock_exists):
        """Test checkpoint_restore() handles missing checkpoint."""
        mock_exists.return_value = False

        repl = REPLEnvironment(context="test")
        result = repl.execute("print(checkpoint_restore('missing_ckpt'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "not found" in result.output

    def test_checkpoint_restore_increments_exploration(self):
        """Test checkpoint_restore() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("checkpoint_restore('test')")

        assert repl._exploration_calls > initial_calls


class TestRegistryLookup:
    """Test _registry_lookup() / registry_lookup() function."""

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="roles:\n  coder_escalation:\n    model:\n      name: Qwen2.5-Coder-32B\n",
    )
    @patch("yaml.safe_load")
    def test_registry_lookup_success(self, mock_yaml, mock_file):
        """Test registry_lookup() finds value."""
        mock_yaml.return_value = {
            "roles": {"coder_escalation": {"model": {"name": "Qwen2.5-Coder-32B"}}}
        }

        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = registry_lookup('roles.coder_escalation.model.name')
print('Qwen' in output)
""")

        assert result.error is None
        assert "True" in result.output

    @patch("builtins.open", new_callable=mock_open, read_data="roles: {}")
    @patch("yaml.safe_load")
    def test_registry_lookup_missing_key(self, mock_yaml, mock_file):
        """Test registry_lookup() handles missing keys."""
        mock_yaml.return_value = {"roles": {}}

        repl = REPLEnvironment(context="test")
        result = repl.execute("print(registry_lookup('roles.nonexistent.key'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "not found" in result.output

    def test_registry_lookup_increments_exploration(self):
        """Test registry_lookup() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("registry_lookup('test.key')")

        assert repl._exploration_calls > initial_calls


class TestRegistryUpdate:
    """Test _registry_update() / registry_update() function."""

    @patch("orchestration.procedure_registry.ProcedureRegistry")
    def test_registry_update(self, mock_registry_class):
        """Test registry_update() delegates to run_procedure."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_result = Mock()
        mock_result.success = True
        mock_result.procedure_id = "update_registry"
        mock_result.error = None
        mock_result.elapsed_seconds = 0.1
        mock_result.outputs = {}
        mock_result.step_results = [Mock(success=True)]
        mock_registry.execute.return_value = mock_result

        repl = REPLEnvironment(context="test")
        result = repl.execute("output = registry_update('roles.test.enabled', True)")

        assert result.error is None


class TestBenchmarkRun:
    """Test _benchmark_run() / benchmark_run() function."""

    @patch("orchestration.procedure_registry.ProcedureRegistry")
    def test_benchmark_run(self, mock_registry_class):
        """Test benchmark_run() delegates to run_procedure."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_result = Mock()
        mock_result.success = True
        mock_result.procedure_id = "benchmark_model"
        mock_result.error = None
        mock_result.elapsed_seconds = 10.0
        mock_result.outputs = {"tps": 25.5}
        mock_result.step_results = [Mock(success=True)]
        mock_registry.execute.return_value = mock_result

        repl = REPLEnvironment(context="test")
        result = repl.execute("output = benchmark_run('/tmp/models/test.gguf')")

        assert result.error is None


class TestBenchmarkCompare:
    """Test _benchmark_compare() / benchmark_compare() function."""

    def test_benchmark_compare_returns_json(self):
        """Test benchmark_compare() returns JSON output."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = benchmark_compare('model_a', 'model_b')
data = json.loads(output)
# Should have 'models' key even if results not found
print('models' in data)
""")

        assert result.error is None
        assert "True" in result.output

    def test_benchmark_compare_increments_exploration(self):
        """Test benchmark_compare() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("benchmark_compare('model_a', 'model_b')")

        assert repl._exploration_calls > initial_calls


class TestGateRun:
    """Test _gate_run() / gate_run() function."""

    @patch("orchestration.procedure_registry.ProcedureRegistry")
    def test_gate_run_default(self, mock_registry_class):
        """Test gate_run() with default parameters."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_result = Mock()
        mock_result.success = True
        mock_result.procedure_id = "gate_runner"
        mock_result.error = None
        mock_result.elapsed_seconds = 0.5
        mock_result.outputs = {"gates_passed": 2, "gates_failed": 0}
        mock_result.step_results = [Mock(success=True)]
        mock_registry.execute.return_value = mock_result

        repl = REPLEnvironment(context="test")
        result = repl.execute("output = gate_run()")

        assert result.error is None

    @patch("orchestration.procedure_registry.ProcedureRegistry")
    def test_gate_run_with_options(self, mock_registry_class):
        """Test gate_run() with specific gates and fix mode."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_result = Mock()
        mock_result.success = True
        mock_result.procedure_id = "gate_runner"
        mock_result.error = None
        mock_result.elapsed_seconds = 1.0
        mock_result.outputs = {"gates_passed": 1, "gates_failed": 0}
        mock_result.step_results = [Mock(success=True)]
        mock_registry.execute.return_value = mock_result

        repl = REPLEnvironment(context="test")
        result = repl.execute("output = gate_run(gates=['lint'], path='src/', fix=True)")

        assert result.error is None

    def test_gate_run_increments_exploration(self):
        """Test gate_run() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("gate_run()")

        assert repl._exploration_calls > initial_calls

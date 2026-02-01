"""Routing, delegation, and episodic memory tools for the REPL environment.

Provides mixin with: escalate, my_role, route_advice, delegate, recall.
"""

from __future__ import annotations

from typing import Any

from src.repl_environment.types import wrap_tool_output


class _RoutingMixin:
    """Mixin providing routing and delegation tools for REPLEnvironment.

    Expects the following attributes from the concrete class:
    - config: REPLConfig
    - context: str
    - artifacts: dict
    - role: str
    - _exploration_calls: int
    - _exploration_log: ExplorationLog
    - llm_primitives: Any | None
    - _retriever: Any | None
    - _hybrid_router: Any | None
    """

    def _recall(self, query: str, limit: int = 5) -> str:
        """Search episodic memory for similar past tasks with Q-values.

        Args:
            query: Natural language description of what you're looking for.
            limit: Maximum number of results to return (default 5).

        Returns:
            JSON string with similar past tasks and routing advice.
        """
        self._exploration_calls += 1
        import json

        # Use retriever if available (reuses shared MemRL components)
        if self._retriever is not None:
            return self._recall_via_retriever(query, limit)

        # Legacy fallback: create fresh instances
        return self._recall_legacy(query, limit)

    def _recall_via_retriever(self, query: str, limit: int = 5) -> str:
        """Recall using shared TwoPhaseRetriever (fast, Q-value aware)."""
        import json

        try:
            results_raw = self._retriever.retrieve_for_exploration(
                query=query,
                context_preview=self.context[:500] if self.context else "",
            )

            results = []
            for r in results_raw[:limit]:
                ctx = r.memory.context or {}
                results.append({
                    "task": ctx.get("objective", r.memory.action)[:200],
                    "outcome": r.memory.outcome or "pending",
                    "action": r.memory.action[:100],
                    "q_value": round(r.q_value, 3),
                    "similarity": round(r.similarity, 3),
                    "combined_score": round(r.combined_score, 3),
                    "role_used": ctx.get("role", "unknown"),
                })

            # Get best action recommendation
            best = self._retriever.get_best_action(results_raw)
            # Handle both base (action, conf) and graph-enhanced (action, conf, warnings)
            if best is not None:
                best_action = best[0]
                best_conf = round(best[1], 3)
            else:
                best_action = None
                best_conf = None

            response = {
                "results": results,
                "best_action": best_action,
                "confidence": best_conf,
            }

            self._exploration_log.add_event("recall", {"query": query}, response)

            if self.config.use_toon_encoding and len(results) >= 3:
                from src.services.toon_encoder import encode
                output = encode(response)
            else:
                output = json.dumps(response, indent=2)
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

        except Exception as e:
            output = json.dumps({"results": [], "best_action": None, "confidence": None, "error": str(e)})
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

    def _recall_legacy(self, query: str, limit: int = 5) -> str:
        """Legacy recall using fresh EpisodicStore + TaskEmbedder."""
        import json

        try:
            from orchestration.repl_memory.episodic_store import EpisodicStore
            from orchestration.repl_memory.embedder import TaskEmbedder
        except ImportError:
            return json.dumps({"results": [], "best_action": None, "confidence": None, "error": "Episodic memory not available"})

        try:
            store = EpisodicStore()
            embedder = TaskEmbedder()

            query_embedding = embedder.embed(query)
            memories = store.search_similar(
                embedding=query_embedding,
                limit=limit,
                min_similarity=0.3,
            )

            results = []
            for mem in memories:
                results.append({
                    "task": mem.task_description[:200] if mem.task_description else "",
                    "outcome": mem.outcome,
                    "action": mem.action[:100] if hasattr(mem, "action") else "unknown",
                    "q_value": round(mem.q_value, 3) if hasattr(mem, "q_value") else 0.5,
                    "similarity": round(mem.similarity, 3) if hasattr(mem, "similarity") else 0.0,
                    "combined_score": 0.0,
                    "role_used": mem.context.get("role", "unknown") if mem.context else "unknown",
                })

            response = {"results": results, "best_action": None, "confidence": None}
            self._exploration_log.add_event("recall", {"query": query}, response)

            if self.config.use_toon_encoding and len(results) >= 3:
                from src.services.toon_encoder import encode
                output = encode(response)
            else:
                output = json.dumps(response, indent=2)
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

        except Exception as e:
            output = json.dumps({"results": [], "best_action": None, "confidence": None, "error": str(e)})
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

    def _escalate(self, reason: str, target_role: str | None = None) -> str:
        """Request escalation to a higher-tier or specific model.

        Args:
            reason: Why escalation is needed (be specific).
            target_role: Optional specific role to escalate to.

        Returns:
            Acknowledgment message.
        """
        self.artifacts["_escalation_requested"] = True
        self.artifacts["_escalation_reason"] = reason
        if target_role:
            # Resolve role aliases (e.g., "reviewer_agent" -> "architect_general")
            target_role = self._resolve_role_alias(target_role)
            self.artifacts["_escalation_target"] = target_role

        target_desc = target_role or "next-tier"
        return f"[ESCALATION REQUESTED -> {target_desc}]: {reason}"

    def _my_role(self) -> str:
        """Get information about the current model's role and capabilities.

        Returns:
            JSON string with role metadata.
        """
        import json
        from src.roles import Role, Tier, get_tier, get_escalation_chain

        role_enum = Role.from_string(self.role)
        try:
            tier = get_tier(self.role)
            tier_val = tier.value
        except Exception:
            tier_val = "C"

        TIER_CAPABILITIES = {
            "A": ["chat", "routing", "intent_classification", "delegation"],
            "B": ["specialized_execution", "complex_reasoning", "delegation"],
            "C": ["parallel_execution", "file_level_tasks"],
            "D": ["draft_speculation"],
        }

        # Workers that Tier A/B can delegate to
        WORKER_ROLES = [
            "worker_general", "worker_math",
            "worker_summarize", "worker_vision",
        ]

        delegate_targets: list[str] = []
        if tier_val in ("A", "B"):
            delegate_targets = list(WORKER_ROLES)
            # Tier A can also delegate to coder
            if tier_val == "A":
                delegate_targets.append("coder_primary")

        # Get escalation target
        chain = get_escalation_chain(self.role)
        escalation_target = str(chain[1]) if len(chain) > 1 else None

        result = {
            "role": self.role,
            "tier": tier_val,
            "capabilities": TIER_CAPABILITIES.get(tier_val, []),
            "escalates_to": escalation_target,
            "can_delegate_to": delegate_targets,
        }

        if self.config.use_toon_encoding:
            from src.services.toon_encoder import encode
            output = encode(result)
        else:
            output = json.dumps(result, indent=2)

        # Track tool output so it can be stripped from answer
        self.artifacts.setdefault("_tool_outputs", []).append(output)
        return wrap_tool_output(output)

    def _route_advice(self, task_description: str) -> str:
        """Get MemRL-informed routing advice for a task.

        Args:
            task_description: Description of the task to route.

        Returns:
            JSON string with routing recommendation.
        """
        self._exploration_calls += 1
        import json

        if self._hybrid_router is None:
            output = json.dumps({
                "recommended_role": None,
                "confidence": 0.0,
                "strategy": "unavailable",
                "similar_tasks": [],
                "warnings": ["MemRL routing not initialized"],
            })
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

        try:
            task_ir = {
                "task_type": "chat",
                "objective": task_description[:200],
                "priority": "interactive",
            }
            routing_decision, strategy = self._hybrid_router.route(task_ir)

            # Get raw retrieval results for transparency
            results = self._hybrid_router.retriever.retrieve_for_routing(task_ir)

            # Aggregate results by role
            role_stats: dict[str, dict] = {}
            for r in results:
                ctx = r.memory.context or {}
                role = ctx.get("role", r.memory.action)
                if role not in role_stats:
                    role_stats[role] = {"q_sum": 0.0, "count": 0}
                role_stats[role]["q_sum"] += r.q_value
                role_stats[role]["count"] += 1

            similar_tasks = []
            for role, stats in sorted(
                role_stats.items(),
                key=lambda x: x[1]["q_sum"] / max(x[1]["count"], 1),
                reverse=True,
            ):
                similar_tasks.append({
                    "role": role,
                    "q_value": round(stats["q_sum"] / max(stats["count"], 1), 3),
                    "count": stats["count"],
                })

            # Collect warnings from graph-enhanced results
            all_warnings: list[str] = []
            for r in results:
                if hasattr(r, "warnings") and r.warnings:
                    all_warnings.extend(r.warnings)

            # Confidence from best result
            confidence = round(results[0].combined_score, 3) if results else 0.0

            response = {
                "recommended_role": routing_decision[0] if routing_decision else None,
                "confidence": confidence,
                "strategy": strategy,
                "similar_tasks": similar_tasks[:5],
                "warnings": list(set(all_warnings))[:3],
            }

            self._exploration_log.add_event(
                "route_advice", {"task": task_description[:100]}, response,
            )

            if self.config.use_toon_encoding:
                from src.services.toon_encoder import encode
                output = encode(response)
            else:
                output = json.dumps(response, indent=2)
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

        except Exception as e:
            output = json.dumps({
                "recommended_role": None,
                "confidence": 0.0,
                "strategy": "error",
                "similar_tasks": [],
                "warnings": [str(e)],
            })
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

    # Role alias mapping: model-generated role names -> actual backend roles
    _ROLE_ALIASES: dict[str, str] = {
        "researcher_agent": "worker_explore",
        "researcher": "worker_explore",
        "coder_agent": "coder_primary",
        "reviewer_agent": "architect_general",
        "reviewer": "architect_general",
        "math_agent": "worker_math",
        "vision_agent": "worker_vision",
        "summarizer_agent": "worker_summarize",
        "summarizer": "worker_summarize",
        "worker_general": "worker_explore",
    }

    def _resolve_role_alias(self, role: str) -> str:
        """Resolve a model-generated role alias to an actual backend role.

        Args:
            role: Role name (may be an alias like "researcher_agent").

        Returns:
            Resolved role name that exists in the backend config.
        """
        return self._ROLE_ALIASES.get(role, role)

    def _delegate(
        self,
        prompt: str,
        target_role: str = "worker_general",
        reason: str = "",
        persona: str = "",
    ) -> str:
        """Delegate a subtask to a specific role with outcome tracking.

        Args:
            prompt: The task to delegate.
            target_role: Role to delegate to.
            reason: Why this role was chosen (helps MemRL learn).
            persona: Optional persona overlay.

        Returns:
            The delegated model's response, or error message.
        """
        self._exploration_calls += 1
        import json

        # Resolve role aliases (e.g., "researcher_agent" -> "worker_explore")
        target_role = self._resolve_role_alias(target_role)

        # Tier guard: workers cannot delegate to other models
        from src.roles import get_tier, Tier
        try:
            tier = get_tier(self.role)
            if tier == Tier.C:
                return "[ERROR: Workers (Tier C) cannot delegate to other models. Use TOOL() for deterministic tools or FINAL() to return results.]"
        except Exception:
            pass  # Unknown role -- allow delegation

        if self.llm_primitives is None:
            return "[ERROR: No LLM primitives available for delegation]"

        import time
        start = time.perf_counter()

        # Initialize delegation tracking list in artifacts
        if "_delegations" not in self.artifacts:
            self.artifacts["_delegations"] = []

        delegation_record: dict = {
            "from_role": self.role,
            "to_role": target_role,
            "reason": reason,
            "persona": persona,
            "prompt_preview": prompt[:100],
            "timestamp": time.time(),
        }

        try:
            result = self.llm_primitives.llm_call(
                prompt, role=target_role, persona=persona or None,
            )
            elapsed = time.perf_counter() - start

            delegation_record["success"] = True
            delegation_record["elapsed_sec"] = round(elapsed, 3)
            delegation_record["result_len"] = len(result)
            self.artifacts["_delegations"].append(delegation_record)

            self._exploration_log.add_event(
                "delegate",
                {"target_role": target_role, "reason": reason},
                f"[{len(result)} chars in {elapsed:.1f}s]",
            )

            return result

        except Exception as e:
            elapsed = time.perf_counter() - start
            delegation_record["success"] = False
            delegation_record["error"] = str(e)
            delegation_record["elapsed_sec"] = round(elapsed, 3)
            self.artifacts["_delegations"].append(delegation_record)

            return f"[DELEGATION FAILED: {target_role} -> {e}]"

# Unreachable Commit Triage (2026-02-19)

Generated: 2026-02-19T19:36:06+01:00

## Summary
- Preserved unreachable commits: 81
- Patch-equivalent already in main: 19
- Candidate not in main (includes WIP/index/noise): 62
- Known push blockers: secret-scanning ancestor commit 5abe83ddbce3; oversized-file commits ec078e04d1b4, fdda7c2b89dc

## Full Ledger
Format: status | sha | date | subject | flags
- candidate-not-in-main | 0aad9ae91cfc | 2026-02-17 02:27:51 +0000 | index on fix/perf-regression: 47df413 fix: patch critical security anti-patterns (eval RCE, cmd injection, MD5) | noise;
- candidate-not-in-main | 0e856adc2ed5 | 2026-01-29 10:14:53 +0100 | index on main: d1313c3 docs: Update BLOCKED_TASKS.md — fix 8 stale statuses, archive 34 completed items | noise;
- candidate-not-in-main | 15ba5d0ca872 | 2026-02-13 00:15:32 +0100 | index on main: 3a25a97 debug: batch 67 — debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python +1 more | near_empty, repl_no_tools, slow_delegation | noise;
- candidate-not-in-main | 17975953f652 | 2026-02-17 02:48:03 +0000 | feat: add doc drift detector for CLAUDE.md ↔ code sync | -
- in-main-equivalent | 18abd069c2f7 | 2026-02-17 03:47:46 +0100 | debug: batch 14 — livecodebench/leetcode_shortest-unsorted-continuous-subarray, livecodebench/leetcode_shortest-unsorted-continuous-subarray, debugbench/debugbench_pascals-triangle-ii_cpp +2 more | repl_no_tools, slow_delegation, wasteful_delegation | -
- candidate-not-in-main | 192bb9a15f51 | 2026-02-05 12:07:05 +0100 | On main: temp local edits | noise;
- candidate-not-in-main | 1aafbf211d4b | 2026-02-13 00:16:31 +0100 | index on main: 3a25a97 debug: batch 67 — debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python +1 more | near_empty, repl_no_tools, slow_delegation | noise;
- candidate-not-in-main | 1aecbc83c4bb | 2026-02-05 12:05:55 +0100 | On main: temp stash for merge | noise;
- candidate-not-in-main | 2191aff05440 | 2026-02-17 03:59:45 +0000 | index on fix/security-footguns: 47df413 fix: patch critical security anti-patterns (eval RCE, cmd injection, MD5) | noise;
- in-main-equivalent | 22bb502820cd | 2026-02-11 03:10:18 +0100 | index on main: 52381e8 debug: batch 49 — mode_advantage_hard/ma_hard_reason_006, mode_advantage_hard/ma_hard_reason_006, mode_advantage_hard/ma_hard_reason_006 | near_empty | noise;
- candidate-not-in-main | 25d695c429a9 | 2026-02-09 05:21:16 +0100 | index on main: 40ae392 fix(tui): preserve code block styling when opening fence scrolls off | noise;
- candidate-not-in-main | 28abeeb28dd9 | 2026-02-17 03:29:30 +0000 | index on docs/backfill-missing-docs: 83ec2fa docs: backfill missing documentation and package docstrings | noise;
- candidate-not-in-main | 2deadb59dd82 | 2026-02-17 17:20:57 +0100 | WIP on main: 360e8ba debug: batch 88 — simpleqa/simpleqa_general_02087, simpleqa/simpleqa_general_02087, simpleqa/simpleqa_general_02087 +2 more | no_skills_available | noise;
- candidate-not-in-main | 2fdcf14a26ec | 2026-02-13 00:15:32 +0100 | WIP on main: 3a25a97 debug: batch 67 — debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python +1 more | near_empty, repl_no_tools, slow_delegation | noise;
- candidate-not-in-main | 329bde5c9446 | 2026-02-09 05:18:40 +0100 | index on main: 40ae392 fix(tui): preserve code block styling when opening fence scrolls off | noise;
- candidate-not-in-main | 33f492134f27 | 2026-02-05 12:05:55 +0100 | index on main: eb27526 chore: Reset episodic memory to clean 104 canonical seeds | noise;
- candidate-not-in-main | 34be457ca7f3 | 2026-02-17 02:39:31 +0000 | index on fix/security-footguns: 47df413 fix: patch critical security anti-patterns (eval RCE, cmd injection, MD5) | noise;
- candidate-not-in-main | 34e070439a95 | 2026-02-02 13:40:50 +0100 | WIP on main: 54ef355 docs: Update progress, architecture docs for 2015 tests / 67.48% coverage | noise;
- in-main-equivalent | 360e8baefe00 | 2026-02-17 10:25:04 +0100 | debug: batch 88 — simpleqa/simpleqa_general_02087, simpleqa/simpleqa_general_02087, simpleqa/simpleqa_general_02087 +2 more | no_skills_available | -
- candidate-not-in-main | 389344b86577 | 2026-02-17 02:36:30 +0000 | index on fix/perf-regression: f41dd2a debug: batch 9 — debugbench/debugbench_valid-parenthesis-string_python, debugbench/debugbench_valid-parenthesis-string_python, debugbench/debugbench_valid-parenthesis-string_python +1 more | format_violation, no_skills_available, prose_only_code_task, repl_no_tools | noise;
- candidate-not-in-main | 4fe088a845d2 | 2026-02-11 19:30:03 +0100 | WIP on main: 71f633c debug: batch 8 — gpqa/gpqa_Astrophysics_0268, gpqa/gpqa_Inorganic Chemistry_0161, gpqa/gpqa_Inorganic Chemistry_0161 +2 more | excessive_tokens, repl_no_tools | noise;
- candidate-not-in-main | 5395fcb18162 | 2026-01-13 15:12:24 +0100 | docs: Close TTT research - NO-GO after benchmark (2026-01-13) | -
- candidate-not-in-main | 56836c8a1462 | 2026-02-02 13:41:13 +0100 | index on main: 54ef355 docs: Update progress, architecture docs for 2015 tests / 67.48% coverage | noise;
- candidate-not-in-main | 5abe83ddbce3 | 2026-02-18 22:25:38 +0100 | Implement middleware hardening trio: credential redaction, script interception, cascading tool policy | secret-scan-blocked;
- candidate-not-in-main | 5ed81e3eb425 | 2026-02-15 14:34:51 +0000 | test: add test_worktree.py for worktree validation | -
- candidate-not-in-main | 6bd7ab4456a0 | 2026-02-17 02:39:31 +0000 | WIP on fix/security-footguns: 47df413 fix: patch critical security anti-patterns (eval RCE, cmd injection, MD5) | noise;
- in-main-equivalent | 7022ac337a8d | 2026-02-17 04:34:02 +0100 | debug: batch 20 — debugbench/debugbench_pascals-triangle-ii_cpp, debugbench/debugbench_pascals-triangle-ii_cpp, debugbench/debugbench_pascals-triangle-ii_cpp +2 more | repl_no_tools | -
- candidate-not-in-main | 7106b24961d4 | 2026-02-17 17:20:57 +0100 | index on main: 360e8ba debug: batch 88 — simpleqa/simpleqa_general_02087, simpleqa/simpleqa_general_02087, simpleqa/simpleqa_general_02087 +2 more | no_skills_available | noise;
- candidate-not-in-main | 714fe057387d | 2026-02-11 03:07:12 +0100 | WIP on main: 52381e8 debug: batch 49 — mode_advantage_hard/ma_hard_reason_006, mode_advantage_hard/ma_hard_reason_006, mode_advantage_hard/ma_hard_reason_006 | near_empty | noise;
- candidate-not-in-main | 753cc700dc5d | 2026-01-29 09:39:55 +0100 | WIP on main: 940697c docs: Update agent tiers, model routing, and escalation paths to match production stack | noise;
- candidate-not-in-main | 7595cb3edf42 | 2026-02-13 14:07:53 +0100 | WIP on main: eab4ff2 feat: orchestrator intelligence improvements (Claude-inspired) | noise;
- candidate-not-in-main | 7953fb3353a0 | 2026-02-13 00:16:31 +0100 | WIP on main: 3a25a97 debug: batch 67 — debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python +1 more | near_empty, repl_no_tools, slow_delegation | noise;
- in-main-equivalent | 79ff030075ab | 2026-02-17 01:32:34 +0100 | debug: batch 2 — usaco/usaco_silver_808_bronze_hoofball, usaco/usaco_silver_808_bronze_hoofball, usaco/usaco_silver_808_bronze_hoofball +2 more | no_skills_available, repl_no_tools | -
- candidate-not-in-main | 7be5187858f9 | 2026-01-13 15:14:00 +0100 | docs: Close TTT research - NO-GO after benchmark (2026-01-13) | -
- candidate-not-in-main | 83b6c8b29baf | 2026-02-17 03:29:30 +0000 | WIP on docs/backfill-missing-docs: 83ec2fa docs: backfill missing documentation and package docstrings | noise;
- in-main-equivalent | 83ec2fa9152b | 2026-02-17 03:21:21 +0000 | docs: backfill missing documentation and package docstrings | -
- candidate-not-in-main | 869e5bd0f3b1 | 2026-02-17 02:36:30 +0000 | WIP on fix/perf-regression: f41dd2a debug: batch 9 — debugbench/debugbench_valid-parenthesis-string_python, debugbench/debugbench_valid-parenthesis-string_python, debugbench/debugbench_valid-parenthesis-string_python +1 more | format_violation, no_skills_available, prose_only_code_task, repl_no_tools | noise;
- candidate-not-in-main | 8d3fbee28bb3 | 2026-02-05 12:07:05 +0100 | index on main: eb27526 chore: Reset episodic memory to clean 104 canonical seeds | noise;
- candidate-not-in-main | 910e411f6751 | 2026-02-09 05:18:40 +0100 | WIP on main: 40ae392 fix(tui): preserve code block styling when opening fence scrolls off | noise;
- in-main-equivalent | 938e6f094bc8 | 2026-02-17 18:34:45 +0100 | docs: final progress report and audit log for disk cleanup session | -
- in-main-equivalent | 940697c33854 | 2026-01-29 09:39:34 +0100 | docs: Update agent tiers, model routing, and escalation paths to match production stack | -
- candidate-not-in-main | 940ed71f9ec7 | 2026-02-07 16:38:52 +0100 | WIP on main: a3bc497 Fix embedder transition to BGE and harden memrl infra | noise;
- candidate-not-in-main | 9629b985934c | 2026-02-17 03:59:45 +0000 | WIP on fix/security-footguns: 47df413 fix: patch critical security anti-patterns (eval RCE, cmd injection, MD5) | noise;
- candidate-not-in-main | 97548ae596ac | 2026-02-17 02:27:51 +0000 | WIP on fix/perf-regression: 47df413 fix: patch critical security anti-patterns (eval RCE, cmd injection, MD5) | noise;
- in-main-equivalent | 98496fa2fc65 | 2026-02-17 05:29:18 +0100 | debug: batch 27 — math/gsm8k_00395, math/gsm8k_00395, math/gsm8k_00395 +2 more | no_skills_available, repl_no_tools | -
- candidate-not-in-main | 9a773afb0da9 | 2026-02-09 05:21:16 +0100 | WIP on main: 40ae392 fix(tui): preserve code block styling when opening fence scrolls off | noise;
- in-main-equivalent | 9ad43479625d | 2026-02-17 07:11:17 +0100 | debug: batch 50 — vl/vl_ocr_0292, vl/vl_ocr_0292, vl/vl_ocr_0292 | no_skills_available, repl_no_tools, vision_blindness | -
- candidate-not-in-main | 9c23333524dc | 2026-01-29 09:40:38 +0100 | index on main: 31dfafe docs: Update agent tiers, model routing, and escalation paths to match production stack | noise;
- candidate-not-in-main | 9c8e39d9da5c | 2026-02-02 13:40:50 +0100 | index on main: 54ef355 docs: Update progress, architecture docs for 2015 tests / 67.48% coverage | noise;
- in-main-equivalent | 9cb7c78cbfff | 2026-02-11 03:10:18 +0100 | On main: Stash unstaged changes during KV fix commit | noise;
- candidate-not-in-main | 9cfd438e84c1 | 2026-02-13 00:27:37 +0100 | WIP on main: 3a25a97 debug: batch 67 — debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python +1 more | near_empty, repl_no_tools, slow_delegation | noise;
- in-main-equivalent | 9d948f8ec78b | 2026-02-15 13:44:05 +0100 | docs: update Ch07 prompt lookup with corpus augmentation, fix spec+lookup compat | -
- in-main-equivalent | a3d72a58340d | 2026-02-17 03:52:53 +0000 | fix: remove stale references from agent skills and governance files | -
- candidate-not-in-main | a5aaad7cc40a | 2026-02-13 14:07:53 +0100 | index on main: eab4ff2 feat: orchestrator intelligence improvements (Claude-inspired) | noise;
- candidate-not-in-main | a82e61bd4061 | 2026-02-02 13:41:13 +0100 | WIP on main: 54ef355 docs: Update progress, architecture docs for 2015 tests / 67.48% coverage | noise;
- in-main-equivalent | aae5b71012af | 2026-02-17 02:38:16 +0000 | perf: fix 6 performance anti-patterns across 4 files | -
- candidate-not-in-main | ac7146732cad | 2026-02-08 16:04:56 +0100 | index on main: 4af4ab6 feat: concept integration (8 features) + streaming inference tap | noise;
- in-main-equivalent | af0986e2cade | 2026-02-17 18:29:59 +0100 | docs: update progress report with corpus build outcome and model cleanup | -
- candidate-not-in-main | b56ae5bd52f3 | 2026-02-07 16:38:52 +0100 | index on main: a3bc497 Fix embedder transition to BGE and harden memrl infra | noise;
- candidate-not-in-main | b6334fe2ba74 | 2026-01-29 10:14:53 +0100 | WIP on main: d1313c3 docs: Update BLOCKED_TASKS.md — fix 8 stale statuses, archive 34 completed items | noise;
- candidate-not-in-main | b8d22f17f836 | 2026-02-17 02:28:28 +0000 | perf: fix 6 performance anti-patterns across 4 files | -
- in-main-equivalent | bef4451c1e48 | 2026-02-17 02:20:23 +0100 | debug: batch 4 — livecodebench/leetcode_number-of-subarrays-with-lcm-equal-to-k, livecodebench/leetcode_find-k-th-smallest-pair-distance, livecodebench/leetcode_find-k-th-smallest-pair-distance +2 more | repl_no_tools, slow_delegation | -
- candidate-not-in-main | c13077316835 | 2026-02-09 05:22:45 +0100 | index on main: 40ae392 fix(tui): preserve code block styling when opening fence scrolls off | noise;
- candidate-not-in-main | c2248993f019 | 2026-02-11 03:07:12 +0100 | index on main: 52381e8 debug: batch 49 — mode_advantage_hard/ma_hard_reason_006, mode_advantage_hard/ma_hard_reason_006, mode_advantage_hard/ma_hard_reason_006 | near_empty | noise;
- candidate-not-in-main | ca1b651d1a18 | 2026-02-17 18:37:18 +0100 | WIP on nightshift/test-gap-coverage: a8ada81 test: add unit tests for 6 critical modules with zero coverage | noise;
- candidate-not-in-main | d420b2ed2ef9 | 2026-02-13 00:27:37 +0100 | index on main: 3a25a97 debug: batch 67 — debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python, debugbench/debugbench_partition-string-into-substrings-with-values-at-most-k_python +1 more | near_empty, repl_no_tools, slow_delegation | noise;
- candidate-not-in-main | d5ac66774399 | 2026-01-14 23:31:39 +0100 | feat: Complete RLM orchestrator + MemRL integration | -
- in-main-equivalent | da8b1749af5a | 2026-02-17 03:13:40 +0000 | docs: backfill missing documentation and package docstrings | -
- candidate-not-in-main | dcc7ba434ccb | 2026-02-09 05:22:45 +0100 | WIP on main: 40ae392 fix(tui): preserve code block styling when opening fence scrolls off | noise;
- candidate-not-in-main | e3fa7baceb05 | 2026-02-15 12:54:40 +0100 | test: add nightshift write test file | -
- candidate-not-in-main | e594e026a1bf | 2026-02-17 02:39:31 +0000 | untracked files on fix/security-footguns: 47df413 fix: patch critical security anti-patterns (eval RCE, cmd injection, MD5) | noise;
- candidate-not-in-main | e9ab8e0601ca | 2026-01-29 09:40:38 +0100 | WIP on main: 31dfafe docs: Update agent tiers, model routing, and escalation paths to match production stack | noise;
- candidate-not-in-main | e9d1dec0a51d | 2026-02-11 19:30:03 +0100 | index on main: 71f633c debug: batch 8 — gpqa/gpqa_Astrophysics_0268, gpqa/gpqa_Inorganic Chemistry_0161, gpqa/gpqa_Inorganic Chemistry_0161 +2 more | excessive_tokens, repl_no_tools | noise;
- candidate-not-in-main | ec078e04d1b4 | 2026-02-15 00:35:26 +0100 | feat: SkillBank core + teacher CLI integration + agent governance refactor | oversize-object-risk;
- candidate-not-in-main | ec2b7cf3e220 | 2026-02-08 16:04:56 +0100 | WIP on main: 4af4ab6 feat: concept integration (8 features) + streaming inference tap | noise;
- candidate-not-in-main | f0abbeb49b14 | 2026-02-05 11:20:26 +0000 | index on main: 1b808ae refactor: Unify code patterns across 125 Python files | noise;
- in-main-equivalent | f41dd2a0b2ea | 2026-02-17 03:06:21 +0100 | debug: batch 9 — debugbench/debugbench_valid-parenthesis-string_python, debugbench/debugbench_valid-parenthesis-string_python, debugbench/debugbench_valid-parenthesis-string_python +1 more | format_violation, no_skills_available, prose_only_code_task, repl_no_tools | -
- candidate-not-in-main | f639fad8bf78 | 2026-02-17 18:37:18 +0100 | index on nightshift/test-gap-coverage: a8ada81 test: add unit tests for 6 critical modules with zero coverage | noise;
- in-main-equivalent | f9a123e2acb0 | 2026-02-17 18:34:05 +0100 | fix(registry): deprecate deleted models and add to deprecated_models list | -
- candidate-not-in-main | fdda7c2b89dc | 2026-01-14 23:25:38 +0100 | feat: Complete RLM orchestrator + MemRL integration | oversize-object-risk;
- candidate-not-in-main | ffc5797cf0ed | 2026-01-29 09:39:55 +0100 | index on main: 940697c docs: Update agent tiers, model routing, and escalation paths to match production stack | noise;

## Notes
- Local preservation refs remain: recovery/unreachable/20260219_192805/* and recovery/stash/20260219_192805/*.
- main and recovery/preserve-20260219_192805 are pushed.
- Bulk push of recovery tags was blocked by GitHub repo rules due to historical objects.

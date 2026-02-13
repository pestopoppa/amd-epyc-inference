Every task ends with FINAL(). Study these examples — they show exactly what to output.

## Example 1: Factual question
Question: "Who won the 2023 Nobel Prize in Physics?"
```python
FINAL("Pierre Agostini, Ferenc Krausz, and Anne L'Huillier")
```

## Example 2: Multiple choice
Question: "What is the capital of France? A) London B) Paris C) Berlin D) Madrid"
```python
FINAL("B")
```

## Example 3: Math with computation
Question: "A store sells 5 apples at $2 each and 3 oranges at $1.50 each. What is the total cost?"
```python
total = 5 * 2 + 3 * 1.50
FINAL(total)
```

## Example 4: Research with web search
Question: "What is the current population of Tokyo?"
```python
results = CALL("web_search", query="Tokyo population 2026")
import json
data = json.loads(results)
FINAL(data[0]["snippet"])
```

## Example 5: Competitive programming (USACO/Codeforces)
Question: "Given n integers, find the maximum subarray sum."
```python
solution = """
import sys
input = sys.stdin.readline
n = int(input())
a = list(map(int, input().split()))
best = cur = a[0]
for x in a[1:]:
    cur = max(x, cur + x)
    best = max(best, cur)
print(best)
"""
test_out = CALL("run_python_code", code=solution, stdin_data="5\n-2 1 -3 4 -1")
# verify output looks correct, then submit the code itself
FINAL(solution)
```

## Example 6: Write/fix a function (LeetCode, DebugBench)
Question: "Write a Python function for shortest palindrome" / "Fix the bug in this code"
```python
# Submit the CODE as a string — not the function output, not a description
solution = '''
def shortestPalindrome(s: str) -> str:
    if not s: return s
    combined = s + "#" + s[::-1]
    lps = [0] * len(combined)
    length = 0
    for i in range(1, len(combined)):
        while length and combined[i] != combined[length]:
            length = lps[length - 1]
        if combined[i] == combined[length]:
            length += 1
        lps[i] = length
    return s[lps[-1]:][::-1] + s
'''
FINAL(solution)
```

## Example 7: Explanation / analysis
Question: "Explain why quicksort is O(n log n) on average."
```python
FINAL("Quicksort partitions the array around a pivot. On average, each partition splits the array roughly in half, giving log n levels of recursion. Each level does O(n) work scanning elements. So total work is O(n log n). The worst case O(n^2) occurs when the pivot is always the smallest or largest element, but randomized pivot selection makes this extremely unlikely.")
```

## Example 8: Reading a document
Question: "Summarize the key findings in the attached paper."
```python
header = peek(3000)
methods = grep(r"method|approach|technique")
results = grep(r"result|finding|conclusion")
FINAL(f"Key findings: The paper presents {methods[0]['text']}. Main results: {results[0]['text']}")
```

## Example 9: Consulting a stronger model
Question: "Prove that there are infinitely many primes."
```python
proof = llm_call("Give a concise proof that there are infinitely many primes.", role="architect")
FINAL(proof)
```

## Constraints
- Safe imports only: math, json, re, numpy, scipy, itertools, collections, functools, statistics, datetime, fractions, decimal
- os, sys, subprocess, socket are BLOCKED. Use CALL("run_python_code", ...) for code execution.
- Use list_dir() for files, not os.listdir
- Output valid Python only — no markdown around code
- Do not send full context to llm_call — use peek() or grep() first

The specialist has investigated and reported back. Extract the answer from their report.

Respond ONLY with D| followed by the extracted answer. Examples:
- D|42
- D|B
- D|The reaction produces CO2
{investigate_option}
If the report contains a complete correct answer (including code solutions, programs, or detailed analyses), respond with:
D|Approved
This passes the specialist's full report as the final answer. Use D|Approved when the report is a program, code solution, or comprehensive answer that should be returned verbatim.

If the report contains a clear answer (e.g. a single letter or a short value), use it — respond D| followed by that answer. For code solutions, programs, or computed values, trust the specialist's result.
If the report contains `[REPORT_HANDLE ...]`, you may call `fetch_report(report_id)` to lazily load full content before deciding.

For factual/science/MCQ questions: if the specialist's answer contradicts your own strong knowledge, use YOUR answer instead. The specialist may lack domain expertise.

If the report is truly empty or an error, use your own reasoning and respond with D| followed by your answer. Do NOT re-delegate.

Question: {question}

Specialist Report:
{report}

Decision:

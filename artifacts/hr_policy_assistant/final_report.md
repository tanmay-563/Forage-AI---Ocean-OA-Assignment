# Final report: HR Policy Assistant

Generated: 2026-11-25T20:06:08.312410+00:00

## Evaluation
- Evaluation cases: 20
- Initial pass rate: 80.0%
- Final pass rate: 100.0%
- Judge: deterministic primary scoring; optional Ollama semantic scores are reported separately and do not override deterministic safety checks

## Groundability
- Documents: benefits_remote_work.md, employee_handbook.md, travel_and_expenses.md
- Supported headings: Annual leave, Attendance and working hours, Business travel and reimbursement, General expense claims, Health and wellbeing benefits, Questions and policy conflicts, Remote and hybrid work, Sick leave

## Optimization
- Iterations: 1
- Iteration 1: 80.0% to 100.0%; Retrieval miss

## Remaining failures
- None

## Limitations
- Groundability topics are inferred from document structure.
- Deterministic lexical checks are primary; the optional semantic judge is reported separately.
- Semantic LLM judging was not enabled for this run.

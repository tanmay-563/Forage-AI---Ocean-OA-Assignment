# ForgeAI demo script (5–7 minutes)

## 1. Brief and documents (45 seconds)
Show the HR handbook in `demos/hr_policy/documents/`. Run:

```powershell
python -m forgeai.cli build --brief "Create an HR policy assistant. Answer documented leave, attendance, benefits, remote work, travel and expense questions, and refuse unsupported questions." --documents demos/hr_policy/documents --top-k 1
```

Explain that the same command and package are used for each domain.

## 2. Intake and groundability (45 seconds)
Open `artifacts/hr_policy_assistant/requirements.json` and `groundability.json`. Show listed sources, inferred topic headings, and conservative assumptions. Explain that heading coverage is a useful signal, not proof every subtopic is supported.

## 3. Definition and evidence (45 seconds)
Open `agent_definition.json` and point to the system instructions, refusal policy, retrieval settings, tools, citation contract, and document collection.

## 4. Evaluation and optimization (90 seconds)
Open `evals/cases.json`, `initial_evaluation.json`, and `optimization_logs.json`. The demo starts with `--top-k 1` to make retrieval misses observable; the optimizer may raise the candidate count based on those failures. Explain the deterministic scoring dimensions and show actual case counts, pass rates, any changes, and remaining failures. Do not imply that lexical checks are a semantic LLM judge.

## 5. Generic agent runner (60 seconds)
Run `python -m forgeai.cli chat --agent artifacts/hr_policy_assistant/agent_definition.json`. Ask: “How much annual leave is provided?” Then ask: “What is the maternity benefit amount?” Show source citations for the supported question and an evidence-based refusal where information is absent.

## 6. Marine procurement (60 seconds)
Run the procurement build command from README. Show catalogue.csv, the two supplier PDFs, the supplier profiles and product specifications. Use `python -m forgeai.cli tools --documents demos/marine_procurement/documents` to show real registered tools. Point out that stock is a dated snapshot and must be reconfirmed.

## 7. Generalization and limits (45 seconds)
Build the laboratory safety assistant with the third README command. Finish by showing `final_report.md` and `KNOWN_LIMITATIONS.md`. State clearly that models are optional, no model was downloaded, and the no-key default is extractive.

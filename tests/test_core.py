import json
from pathlib import Path
import pytest
from forgeai.evaluation.cases import generate_cases
from forgeai.evaluation.judge import evaluate, score
from forgeai.evaluation.optimizer import optimize
from forgeai.groundability import analyze
from forgeai.ingestion import ingest
from forgeai.intake import intake
from forgeai.models import AgentDefinition, Chunk, EvalCase
from forgeai.providers import ExtractiveProvider
from forgeai.generator import generate
from forgeai.runner import AgentRunner, from_definition
from forgeai.tools.registry import build_registry, csv_lookup, safe_calculate


def test_ingest_text_markdown_and_chunking(tmp_path):
    (tmp_path / "policy.md").write_text("# Policy\n## Leave\nAnnual leave is 18 days each year.\nMore details here.", encoding="utf-8")
    chunks = ingest(tmp_path, chunk_chars=35, overlap=4)
    assert chunks and all(c.source == "policy.md" for c in chunks)
    assert any(c.section == "Leave" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_csv_ingest_has_row_references(tmp_path):
    (tmp_path / "items.csv").write_text("code,price\nA,10\nB,20\n", encoding="utf-8")
    chunks = ingest(tmp_path)
    assert [chunk.row for chunk in chunks] == [2, 3]
    assert "code: A" in chunks[0].text


def test_pdf_ingest_and_page_metadata(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Supplier quote valid for fourteen calendar days.")
    doc.save(tmp_path / "supplier.pdf")
    doc.close()
    chunks = ingest(tmp_path)
    assert chunks[0].page == 1
    assert "fourteen calendar days" in chunks[0].text


def test_groundability_topics_and_retrieval():
    chunks = [Chunk(chunk_id="one", source="policy.md", text="## Annual leave\nEmployees receive annual leave.", section="Annual leave")]
    assert analyze(chunks)["supported_topics"] == ["Annual leave"]
    from forgeai.retrieval import Retriever
    assert Retriever(chunks).search("annual leave")
    assert not Retriever(chunks).search("orbital engine warranty")


def test_intake_asks_at_most_five_and_records_unattended(tmp_path):
    (tmp_path / "policy.md").write_text("# Policy\nA sufficiently detailed employee policy is present.", encoding="utf-8")
    req, questions = intake("Create an assistant for employee policy", tmp_path, unattended=False)
    assert len(questions) <= 5
    unattended, no_questions = intake("Create an assistant for employee policy", tmp_path, unattended=True)
    assert not no_questions and unattended.assumptions
    assert req.agent_id


def test_agent_definition_schema_rejects_bad_id(tmp_path):
    chunks = [Chunk(chunk_id="x", source="x.md", text="Policy says annual leave is 18 days.")]
    from forgeai.models import RequirementDocument
    req = RequirementDocument(agent_id="x", agent_name="X", purpose="Answer questions", audience="staff", scope="docs", supported_topics=[], unsupported_topics=[], tone="plain", must_answer=[], must_refuse=[], output_format="answer", available_documents=["x.md"], required_tools=[])
    assert generate(req, str(tmp_path)).retrieval["method"] == "local_lexical"
    data = generate(req, str(tmp_path)).model_dump(); data["agent_id"] = "../oops"
    with pytest.raises(ValueError): AgentDefinition.model_validate(data)


def test_tools_calculator_safety():
    assert safe_calculate("(12 + 8) * 3") == 60
    for bad in ("__import__('os')", "2 ** 100", "1 / 0"):
        with pytest.raises(ValueError): safe_calculate(bad)


def test_csv_lookup_restricts_root_and_returns_rows(tmp_path):
    (tmp_path / "catalog.csv").write_text("code,description\nA,blue valve\nB,red pump\n", encoding="utf-8")
    assert csv_lookup(tmp_path, "catalog.csv", "blue")[0]["row"] == 2
    with pytest.raises(ValueError): csv_lookup(tmp_path, "../secret.csv", "blue")
    with pytest.raises(ValueError): csv_lookup(tmp_path, "catalog.csv", "")


def test_registry_validates_schema_and_tools():
    registry_root = Path(pytest.ensuretemp("registry")) if hasattr(pytest, "ensuretemp") else None
    # A CSV lookup is available only when an approved document collection is supplied.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        registry = build_registry(Path(temp))
        assert len(registry.list()) == 4
        assert registry.execute("calculator", {"expression": "2+3"}) == 5
        with pytest.raises(ValueError): registry.execute("calculator", {"code": "2+3"})
        with pytest.raises(ValueError): registry.execute("calculator", {"expression": "2+3", "extra": 1})


def test_weather_tool_validates_and_calls_public_api(monkeypatch):
    import forgeai.tools.registry as module
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"current": {"temperature_2m": 20}}
    calls = []
    monkeypatch.setattr(module.httpx, "get", lambda *a, **kw: calls.append((a, kw)) or Response())
    result = build_registry().execute("weather_forecast", {"latitude": 10, "longitude": 20})
    assert result["temperature_2m"] == 20 and calls
    with pytest.raises(ValueError): build_registry().execute("weather_forecast", {"latitude": 100, "longitude": 0})


def test_semantic_judge_uses_validated_mocked_response(monkeypatch):
    from forgeai.evaluation.semantic_judge import DIMENSIONS, OllamaSemanticJudge
    import forgeai.evaluation.semantic_judge as module
    payload = {"scores": {name: 0.8 for name in DIMENSIONS}, "notes": ["Evidence supports the answer."]}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": json.dumps(payload)}}
    monkeypatch.setattr(module.httpx, "post", lambda *args, **kwargs: Response())
    case = EvalCase(test_id="x", question="Q", expected_behavior="answer", expected_facts=[], category="test", answerable=True, refusal_expected=False)
    result = OllamaSemanticJudge("http://localhost:11434", "test-model")(case, {"answer": "A", "sources": [], "evidence": []})
    assert result["scores"]["correctness"] == 0.8


def test_semantic_judge_malformed_response_is_recorded(monkeypatch):
    from forgeai.evaluation.semantic_judge import OllamaSemanticJudge
    import forgeai.evaluation.semantic_judge as module
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "not JSON"}}
    monkeypatch.setattr(module.httpx, "post", lambda *args, **kwargs: Response())
    case = EvalCase(test_id="bad", question="q", expected_behavior="answer", category="x", answerable=True, refusal_expected=False)
    reviewed = evaluate(case, {"answer": "a", "sources": [], "evidence": []}, OllamaSemanticJudge("http://local", "mock"))
    assert reviewed.semantic_scores is None
    assert "JSONDecodeError" in reviewed.semantic_judge_error


@pytest.mark.parametrize("case,output,expected", [
    (EvalCase(test_id="r", question="q", expected_behavior="answer", category="x", answerable=True, refusal_expected=False), {"answer": "I cannot answer from the available documents.", "sources": [], "evidence": []}, "Retrieval miss"),
    (EvalCase(test_id="w", question="q", expected_behavior="answer", expected_facts=["Annual leave gives 18 days"], category="x", answerable=True, refusal_expected=False), {"answer": "I cannot verify that.", "sources": [{"source":"a.md", "chunk_id":"x"}], "evidence": []}, "Wrong refusal"),
    (EvalCase(test_id="h", question="q", expected_behavior="refuse", category="x", answerable=False, refusal_expected=True), {"answer": "The answer is 42.", "sources": [], "evidence": []}, "Hallucination"),
    (EvalCase(test_id="p", question="q", expected_behavior="answer", expected_facts=["Annual leave gives 18 days"], category="x", answerable=True, refusal_expected=False), {"answer": "A different fact.", "sources": [{"source":"a.md", "chunk_id":"x"}], "evidence": []}, "Prompt defect"),
])
def test_failure_classification_categories(case, output, expected):
    assert evaluate(case, output).failure_classification == expected


def test_case_generator_uses_document_text_and_case_ids_unique():
    chunks = [Chunk(chunk_id="a", source="a.md", text="Annual leave provides eighteen working days to regular full time employees each year.")]
    cases = generate_cases(chunks, minimum=1)
    assert cases[0].expected_facts[0].startswith("Annual leave")
    assert cases[0].expected_facts[0] not in cases[0].question
    assert len({c.test_id for c in cases}) == len(cases)


def test_judge_scoring_and_failure_classification():
    case = EvalCase(test_id="a", question="Q?", expected_behavior="answer", expected_facts=["Annual leave provides eighteen days"], supporting_source="policy.md", category="direct", answerable=True, refusal_expected=False)
    good = evaluate(case, {"answer": "Annual leave provides eighteen days.", "sources": [{"source": "policy.md", "chunk_id": "x"}], "evidence": [{"text": "Annual leave provides eighteen days"}]})
    assert good.passed and score([good])["pass_rate"] == 1
    bad = evaluate(case, {"answer": "I cannot answer from the available documents.", "sources": [], "evidence": []})
    assert bad.failure_classification == "Retrieval miss"


def test_runner_quotes_retrieved_evidence_and_refuses(tmp_path):
    (tmp_path / "policy.md").write_text("# Leave\nAnnual leave is 18 days each year.\n", encoding="utf-8")
    data = {"agent_id": "test", "name": "Test", "version": "1.0", "description": "test", "system_prompt": "Use evidence.", "audience": "staff", "supported_topics": [], "unsupported_topics": [], "refusal_policy": {}, "retrieval": {"top_k": 3, "minimum_score": .1}, "chunking": {}, "embedding": {}, "tools": [], "output_contract": {}, "citation_requirements": {}, "document_collection": str(tmp_path), "metadata": {}}
    definition = AgentDefinition.model_validate(data)
    runner = AgentRunner(definition, tmp_path, ExtractiveProvider())
    answer = runner.answer("How many annual leave days are provided?")
    assert "18 days" in answer["answer"] and answer["sources"]
    unsupported = runner.answer("What is the orbital engine warranty in light years?")
    assert "cannot answer" in unsupported["answer"] and not unsupported["sources"]
    agent_file = tmp_path / "definition.json"; agent_file.write_text(json.dumps(data), encoding="utf-8")
    assert from_definition(agent_file).definition.agent_id == "test"


def test_runner_asks_for_clarification_when_sections_are_ambiguous(tmp_path):
    (tmp_path / "policy.md").write_text("# Handbook\n## Annual leave\nEmployees receive annual leave.\n## Sick leave\nEmployees receive sick leave.", encoding="utf-8")
    data = {"agent_id":"ambiguous", "name":"Policy", "description":"test", "system_prompt":"Use evidence", "audience":"staff", "supported_topics":[], "unsupported_topics":[], "refusal_policy":{}, "retrieval":{"minimum_score":.1}, "chunking":{}, "embedding":{}, "tools":[], "output_contract":{}, "citation_requirements":{}, "document_collection":str(tmp_path), "metadata":{}}
    result = AgentRunner(AgentDefinition.model_validate(data), tmp_path).answer("How much leave is available?")
    assert "Which one do you mean?" in result["answer"]
    assert "Annual leave" in result["answer"] and "Sick leave" in result["answer"]


def test_runner_executes_only_selected_csv_tool(tmp_path):
    (tmp_path / "catalog.csv").write_text("code,description\nHFE-220,Hydraulic filter\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Catalogue\nThe HFE-220 is a hydraulic filter.", encoding="utf-8")
    data = {"agent_id":"tool_agent", "name":"Tool agent", "description":"test", "system_prompt":"Use evidence", "audience":"buyers", "supported_topics":[], "unsupported_topics":[], "refusal_policy":{}, "retrieval":{"minimum_score":.1}, "chunking":{}, "embedding":{}, "tools":[{"name":"csv_lookup", "reason":"catalog"}], "output_contract":{}, "citation_requirements":{}, "document_collection":str(tmp_path), "metadata":{}}
    selected = AgentRunner(AgentDefinition.model_validate(data), tmp_path).answer("Find catalogue product HFE-220")
    assert selected["tool_calls"][0]["tool"] == "csv_lookup"
    assert any(source["row"] == 2 for source in selected["sources"])
    data["tools"] = []
    unselected = AgentRunner(AgentDefinition.model_validate(data), tmp_path).answer("Find catalogue product HFE-220")
    assert unselected["tool_calls"] == []


def test_missing_documents_invalid_inputs_and_malformed_cases(tmp_path):
    with pytest.raises(FileNotFoundError): ingest(tmp_path / "missing")
    with pytest.raises(ValueError):
        AgentRunner(AgentDefinition.model_validate({"agent_id":"ok","name":"x","description":"x","system_prompt":"x","audience":"x","supported_topics":[],"unsupported_topics":[],"refusal_policy":{},"retrieval":{},"chunking":{},"embedding":{},"tools":[],"output_contract":{},"citation_requirements":{},"document_collection":"."}), tmp_path).answer("")
    from forgeai.pipeline import load_cases
    bad = tmp_path / "cases.json"; bad.write_text('[{"test_id":"x"}]', encoding="utf-8")
    with pytest.raises(Exception): load_cases(bad)


def test_optimizer_caps_iterations_and_reuses_cases(tmp_path):
    (tmp_path / "policy.md").write_text("# Leave\nAnnual leave provides eighteen working days for staff each year.\n", encoding="utf-8")
    chunks = ingest(tmp_path)
    from forgeai.models import RequirementDocument
    req = RequirementDocument(agent_id="x", agent_name="X", purpose="answer", audience="staff", scope="docs", supported_topics=[], unsupported_topics=[], tone="plain", must_answer=[], must_refuse=[], output_format="answer", available_documents=["policy.md"], required_tools=[])
    definition = generate(req, str(tmp_path))
    cases = generate_cases(chunks, 1)
    with pytest.raises(ValueError): optimize(definition, chunks, cases, tmp_path, max_iterations=4)
    logs, _ = optimize(definition, chunks, cases, tmp_path, max_iterations=1)
    assert len(logs) <= 1


def test_optimizer_rolls_back_a_candidate_without_improvement(tmp_path):
    (tmp_path / "policy.md").write_text("# Policy\n## Leave\nAnnual leave provides eighteen working days each year.", encoding="utf-8")
    chunks = ingest(tmp_path)
    from forgeai.models import RequirementDocument
    req = RequirementDocument(agent_id="rollback", agent_name="Rollback", purpose="answer", audience="staff", scope="docs", supported_topics=[], unsupported_topics=[], tone="plain", must_answer=[], must_refuse=[], output_format="answer", available_documents=["policy.md"], required_tools=[])
    definition = generate(req, str(tmp_path))
    case = EvalCase(test_id="bad_criteria", question="What is the annual leave amount?", expected_behavior="answer", expected_facts=["The policy defines a blue moon bonus"], supporting_source="policy.md", expected_evidence=["The policy defines a blue moon bonus"], category="direct", answerable=True, refusal_expected=False)
    logs, best = optimize(definition, chunks, [case], tmp_path, max_iterations=1)
    assert len(logs) == 1 and logs[0]["accepted"] is False
    assert logs[0]["rollback_reason"]
    assert best.system_prompt == definition.system_prompt

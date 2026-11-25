import json
from pathlib import Path
import pytest
from forgeai.evaluation.cases import (InsufficientEvidenceError, generate_cases,
                                      generate_cases_with_report, validate_case_diversity,
                                      validate_case_provenance)
from forgeai.evaluation.judge import evaluate, score
from forgeai.evaluation.optimizer import optimize
from forgeai.groundability import analyze
from forgeai.ingestion import diagnose_collection, ingest
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


def test_pdf_text_repairs_split_word_extraction_artifact():
    from forgeai.ingestion import _join_pdf_lines
    assert "available" in _join_pdf_lines("Certificates of conformity are availab le for products.")


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
    row_chunk = next(chunk for chunk in ingest(tmp_path) if chunk.row == 2)
    assert any(source["chunk_id"] == row_chunk.chunk_id for source in selected["sources"] if source["row"] == 2)
    data["tools"] = []
    unselected = AgentRunner(AgentDefinition.model_validate(data), tmp_path).answer("Find catalogue product HFE-220")
    assert unselected["tool_calls"] == []


def _csv_runner(tmp_path, rows, top_k=5):
    (tmp_path / "catalogue.csv").write_text("product_code,unit,currency,unit_price\n" + rows, encoding="utf-8")
    data = {"agent_id":"price_test", "name":"Price test", "description":"test", "system_prompt":"Use evidence", "audience":"buyers", "supported_topics":[], "unsupported_topics":[], "refusal_policy":{}, "retrieval":{"top_k":top_k,"minimum_score":.05}, "chunking":{}, "embedding":{}, "tools":[{"name":"csv_lookup","reason":"catalogue"}], "output_contract":{}, "citation_requirements":{}, "document_collection":str(tmp_path), "metadata":{}}
    return AgentRunner(AgentDefinition.model_validate(data), tmp_path)


def test_grounded_price_difference_uses_verified_rows_and_decimal_formula(tmp_path):
    runner = _csv_runner(tmp_path, "PUMP-10,each,USD,84.00\nPUMP-20,each,USD,112.50\n")
    result = runner.answer("Which item has the lower price, PUMP-10 or PUMP-20, and what is the price difference?")
    assert result["calculation"] == {"status":"calculated", "formula":"|84.00 USD/each - 112.50 USD/each| = 28.50 USD/each", "result":"28.50", "currency":"USD", "unit":"each"}
    assert "PUMP-10" in result["answer"] and "28.50 USD per each" in result["answer"]
    assert len(result["sources"]) == 2 and all(source["row"] in {2, 3} for source in result["sources"])
    chunks = {chunk.chunk_id: chunk for chunk in runner.chunks}
    assert all(source["chunk_id"] in chunks and chunks[source["chunk_id"]].source == source["source"] for source in result["sources"])
    operands = result["tool_calls"][-1]["operands"]
    assert all(operand["quotation"] in chunks[operand["chunk_id"]].text for operand in operands)


def test_price_difference_refuses_missing_operand(tmp_path):
    runner = _csv_runner(tmp_path, "PUMP-10,each,USD,84.00\n")
    result = runner.answer("Compare prices for PUMP-10 and PUMP-20; what is the price difference?")
    assert result["calculation"]["status"] == "refused"
    assert "missing" in result["answer"]


@pytest.mark.parametrize(("rows", "reason"), [
    ("PUMP-10,each,USD,84.00\nPUMP-20,each,EUR,112.50\n", "currencies"),
    ("PUMP-10,each,USD,84.00\nPUMP-20,kit,USD,112.50\n", "units"),
])
def test_price_difference_refuses_incompatible_currency_or_unit(tmp_path, rows, reason):
    runner = _csv_runner(tmp_path, rows)
    result = runner.answer("Compare prices for PUMP-10 and PUMP-20; what is the price difference?")
    assert result["calculation"]["status"] == "refused"
    assert reason in result["calculation"]["reason"]


def test_cross_document_question_retrieves_both_relevant_sources(tmp_path):
    (tmp_path / "catalogue.md").write_text("The catalogue currency code for PUMP-10 is USD.", encoding="utf-8")
    (tmp_path / "policy.md").write_text("Buyers must preserve the currency code when comparing catalogue entries.", encoding="utf-8")
    data = {"agent_id":"cross_doc", "name":"Cross doc", "description":"test", "system_prompt":"Use evidence", "audience":"buyers", "supported_topics":[], "unsupported_topics":[], "refusal_policy":{}, "retrieval":{"top_k":3,"minimum_score":.05}, "chunking":{}, "embedding":{}, "tools":[], "output_contract":{}, "citation_requirements":{}, "document_collection":str(tmp_path), "metadata":{}}
    result = AgentRunner(AgentDefinition.model_validate(data), tmp_path).answer("What does each document say about the currency code?")
    assert {source["source"] for source in result["sources"]} == {"catalogue.md", "policy.md"}
    assert "catalogue.md" in result["answer"] and "policy.md" in result["answer"]


def test_retrieval_ignores_generic_source_wording_without_lowering_threshold(tmp_path):
    (tmp_path / "travel.md").write_text("Lodging is reimbursed up to INR 6,000 per night outside metro cities and INR 8,000 per night in metro cities.", encoding="utf-8")
    data = {"agent_id":"travel", "name":"Travel", "description":"test", "system_prompt":"Use evidence", "audience":"staff", "supported_topics":[], "unsupported_topics":[], "refusal_policy":{}, "retrieval":{"top_k":3,"minimum_score":.25}, "chunking":{}, "embedding":{}, "tools":[], "output_contract":{}, "citation_requirements":{}, "document_collection":str(tmp_path), "metadata":{}}
    result = AgentRunner(AgentDefinition.model_validate(data), tmp_path).answer("What does the source say about lodging?")
    assert "INR 6,000" in result["answer"] and result["sources"][0]["source"] == "travel.md"


def test_insufficient_query_refuses_without_invented_citations(tmp_path):
    (tmp_path / "policy.md").write_text("The pump warranty is valid for two years.", encoding="utf-8")
    data = {"agent_id":"insufficient", "name":"Insufficient", "description":"test", "system_prompt":"Use evidence", "audience":"buyers", "supported_topics":[], "unsupported_topics":[], "refusal_policy":{}, "retrieval":{"top_k":3,"minimum_score":.1}, "chunking":{}, "embedding":{}, "tools":[], "output_contract":{}, "citation_requirements":{}, "document_collection":str(tmp_path), "metadata":{}}
    result = AgentRunner(AgentDefinition.model_validate(data), tmp_path).answer("What is the certified maximum submersible depth of model Z-900?")
    assert "cannot answer" in result["answer"].casefold()
    assert result["sources"] == []


def test_numeric_conflict_across_documents_is_reported_with_both_sources(tmp_path):
    (tmp_path / "manual.md").write_text("Pump A maximum operating pressure is 50 bar.", encoding="utf-8")
    (tmp_path / "bulletin.md").write_text("Pump A maximum operating pressure is 80 bar.", encoding="utf-8")
    data = {"agent_id":"conflict", "name":"Conflict", "description":"test", "system_prompt":"Use evidence", "audience":"buyers", "supported_topics":[], "unsupported_topics":[], "refusal_policy":{}, "retrieval":{"top_k":4,"minimum_score":.05}, "chunking":{}, "embedding":{}, "tools":[], "output_contract":{}, "citation_requirements":{}, "document_collection":str(tmp_path), "metadata":{}}
    result = AgentRunner(AgentDefinition.model_validate(data), tmp_path).answer("What is Pump A maximum operating pressure?")
    assert result["evidence_status"] == "conflicting"
    assert {source["source"] for source in result["sources"]} == {"manual.md", "bulletin.md"}
    assert "cannot resolve" in result["answer"]


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


def _many_evidence_chunks(count=22):
    return [Chunk(chunk_id=f"chunk-{i}", source=f"source-{i % 3}.md", page=None,
                  text=f"Domain{i} requires interns to analyze dataset{i} records carefully and present findings to reviewers.")
            for i in range(count)]


def test_evidence_case_provenance_is_exact_and_validated():
    chunks = [Chunk(chunk_id="source-1", source="policy.pdf", page=3,
                    text="Employees receive twenty annual leave days after completing one year of service.")]
    case = generate_cases([*chunks, *_many_evidence_chunks()], minimum=20)[0]
    assert case.source_filename == case.supporting_source
    assert case.page_number == 3
    assert case.chunk_id == "source-1"
    assert case.supporting_quotation in chunks[0].text
    assert validate_case_provenance(case, chunks + _many_evidence_chunks()) == []


def test_invalid_page_reference_and_missing_chunk_are_rejected():
    chunks = [Chunk(chunk_id="real", source="policy.pdf", page=2,
                    text="Employees receive twenty annual leave days after completing one year of service.")]
    case = generate_cases([*chunks, *_many_evidence_chunks()], minimum=20)[0].model_copy(update={"page_number": 8})
    assert "page number does not match" in " ".join(validate_case_provenance(case, chunks + _many_evidence_chunks()))
    missing = case.model_copy(update={"chunk_id": "missing"})
    assert "source chunk does not exist" in " ".join(validate_case_provenance(missing, chunks + _many_evidence_chunks()))


def test_generator_reports_twenty_factual_cases_separately_from_refusals():
    cases, report = generate_cases_with_report(_many_evidence_chunks(), minimum=20)
    assert report["valid_evidence_backed_cases"] >= 20
    assert report["valid_unsupported_refusal_cases"] >= 2
    assert sum(case.answerability_classification == "evidence_backed" for case in cases) >= 20
    assert all(case.answerability_classification == "unsupported_refusal" for case in cases[-2:])


def test_insufficient_collection_does_not_count_refusals_toward_minimum():
    with pytest.raises(InsufficientEvidenceError) as error:
        generate_cases_with_report([Chunk(chunk_id="few", source="short.md",
            text="This policy provides fourteen annual leave days for eligible employees each year.")], minimum=20)
    assert error.value.report["valid_evidence_backed_cases"] < 20
    assert error.value.report["valid_unsupported_refusal_cases"] == 2
    assert error.value.report["insufficient_evidence_reasons"]


def test_near_duplicate_questions_and_repeated_facts_are_rejected():
    repeated = "The internship requires applicants to complete careful research using Python and present findings to a mentor."
    chunks = [Chunk(chunk_id="a", source="a.md", text=repeated),
              Chunk(chunk_id="b", source="b.md", text=repeated)]
    cases, report = generate_cases_with_report([*chunks, *_many_evidence_chunks()], minimum=20)
    assert len({case.question.casefold() for case in cases}) == len(cases)
    assert report["rejected_case_count"] >= 1


def test_diagnostics_reports_empty_pdf_pages(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    pdf = fitz.open()
    pdf.new_page()
    page = pdf.new_page()
    page.insert_text((72, 72), "This page contains readable extracted text for a real document fact.")
    pdf.save(tmp_path / "pages.pdf")
    pdf.close()
    diagnostic = diagnose_collection(tmp_path)
    assert diagnostic["documents"][0]["page_count"] == 2
    assert diagnostic["documents"][0]["pages"][0]["extracted_characters"] == 0
    assert diagnostic["documents"][0]["pages"][1]["qualifying_sentences"] == 1


def test_pdf_extraction_failure_is_recorded_in_diagnostics(tmp_path, monkeypatch):
    import forgeai.ingestion as module
    (tmp_path / "broken.pdf").write_bytes(b"not a PDF")
    monkeypatch.setattr(module, "_extract", lambda path: (_ for _ in ()).throw(RuntimeError("extraction failed")))
    result = diagnose_collection(tmp_path)
    assert result["extraction_failures"][0]["source"] == "broken.pdf"
    assert "extraction failed" in result["documents"][0]["extraction_error"]


def test_saved_suite_validation_uses_same_provenance_rules_as_generation(tmp_path):
    from forgeai.pipeline import load_cases
    chunks = _many_evidence_chunks()
    cases = generate_cases(chunks, minimum=20)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case.model_dump() for case in cases]), encoding="utf-8")
    assert load_cases(path, chunks)
    broken = [case.model_dump() for case in cases]
    broken[0]["page_number"] = 99
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid provenance"):
        load_cases(path, chunks)


def test_duplicate_and_near_duplicate_questions_are_rejected():
    one = EvalCase(test_id="one", question="What does the internship say about Python skills?",
        expected_behavior="Answer from the source", expected_facts=["The internship requires Python coding skills and analytical reasoning."],
        source_filename="role.md", chunk_id="a", supporting_quotation="The internship requires Python coding skills and analytical reasoning.",
        evidence_validation_status="validated", category="direct factual", answerable=True, refusal_expected=False)
    exact = one.model_copy(update={"test_id": "exact"})
    near = one.model_copy(update={"test_id": "near", "question": "What is stated about Python coding skills?",
        "expected_facts": ["The internship requires Python coding skills and stronger analytical reasoning."]})
    assert any("duplicate or near-duplicate" in error for error in validate_case_diversity([one, exact]))
    assert any("duplicate or near-duplicate" in error for error in validate_case_diversity([one, near]))


def test_cli_and_streamlit_import_the_shared_build_pipeline():
    import forgeai.pipeline as pipeline
    import forgeai.cli as cli
    import app as streamlit_app
    assert cli.build_agent is pipeline.build_agent
    assert streamlit_app.build_agent is pipeline.build_agent


def test_csv_cases_reference_actual_rows_and_columns():
    chunks = [Chunk(chunk_id="row2", source="parts.csv", row=2, text="code: X7 | material: stainless steel | lead_time: 12 days")]
    cases, _ = generate_cases_with_report([*chunks, *_many_evidence_chunks()], minimum=20)
    csv_case = next(case for case in cases if case.chunk_id == "row2")
    assert csv_case.category == "CSV lookup"
    assert "row 2" in csv_case.question and "material" in csv_case.question
    assert csv_case.expected_facts == ["material: stainless steel"]
    assert validate_case_provenance(csv_case, [*chunks, *_many_evidence_chunks()]) == []


def test_comparison_case_keeps_provenance_for_both_documents():
    chunks = [
        Chunk(chunk_id="a", source="terms.md", text="Supplier payment is due within thirty calendar days after receipt of invoice."),
        Chunk(chunk_id="b", source="quote.md", text="Supplier payment is due within fifteen calendar days after delivery acceptance."),
        *_many_evidence_chunks(),
    ]
    cases, _ = generate_cases_with_report(chunks, minimum=20)
    comparison = next(case for case in cases if case.category == "cross-document reasoning")
    assert len(comparison.evidence_provenance) == 2
    assert {ref.source_filename for ref in comparison.evidence_provenance} == {"terms.md", "quote.md"}
    assert validate_case_provenance(comparison, chunks) == []


def test_csv_numerical_case_computes_and_validates_difference():
    chunks = [
        Chunk(chunk_id="r2", source="catalogue.csv", row=2, text="product_code: A-1 | currency: USD | unit_price: 84.00"),
        Chunk(chunk_id="r3", source="catalogue.csv", row=3, text="product_code: B-2 | currency: USD | unit_price: 112.50"),
        *_many_evidence_chunks(),
    ]
    cases, _ = generate_cases_with_report(chunks, minimum=20)
    numeric = next(case for case in cases if case.category == "numerical reasoning")
    assert "28.50" in numeric.expected_facts[0]
    assert [ref.row_number for ref in numeric.evidence_provenance] == [2, 3]
    assert validate_case_provenance(numeric, chunks) == []
    incorrect = numeric.model_copy(update={"reasoning_metadata": {**numeric.reasoning_metadata, "expected_difference": "29"}})
    assert "numerical result does not match" in " ".join(validate_case_provenance(incorrect, chunks))


def test_blind_judge_swaps_order_and_records_position_sensitive_decision(monkeypatch):
    from forgeai.evaluation.semantic_judge import DIMENSIONS, OllamaSemanticJudge
    import forgeai.evaluation.semantic_judge as module
    payloads = []
    class Response:
        def __init__(self, answer): self.answer = answer
        def raise_for_status(self): pass
        def json(self):
            parsed = json.loads(self.answer)
            return {"message": {"content": json.dumps({"scores": {d: .7 for d in DIMENSIONS}, "preferred": "A", "notes": [], "seen": parsed})}}
    def fake_post(url, json, timeout):
        body = json_module.loads(json["messages"][1]["content"])
        payloads.append(body)
        return Response(json_module.dumps({"ok": True}))
    import json as json_module
    monkeypatch.setattr(module.httpx, "post", fake_post)
    case = EvalCase(test_id="blind", question="Which amount applies?", expected_behavior="answer", expected_facts=["Amount is 12"], category="test", answerable=True, refusal_expected=False)
    reviewed = OllamaSemanticJudge("http://local", "judge", generation_model="writer")(
        case, {"answer": "Amount is 12", "sources": [], "evidence": []})
    assert len(payloads) == 2
    assert set(payloads[0]) == set(payloads[1]) == {"question", "expected_behavior", "evidence", "sources", "answer_A", "answer_B"}
    assert "model" not in payloads[0] and "ForgeAI" not in json.dumps(payloads[0])
    assert payloads[0]["answer_A"] == payloads[1]["answer_B"]
    assert reviewed["order_swapped_result"]["decision_changed"] is True
    assert reviewed["same_model_judging"] is False


def test_same_model_evaluator_is_disclosed():
    from forgeai.evaluation.semantic_judge import OllamaSemanticJudge
    judge = OllamaSemanticJudge("http://local", "same", generation_model="same")
    assert judge.same_model is True


def test_same_model_limitation_is_persisted_in_case_result(monkeypatch):
    import forgeai.evaluation.semantic_judge as module
    from forgeai.evaluation.semantic_judge import DIMENSIONS, OllamaSemanticJudge
    payload = {"scores": {name: .8 for name in DIMENSIONS}, "preferred": "tie", "notes": ["No material difference."]}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": json.dumps(payload)}}
    monkeypatch.setattr(module.httpx, "post", lambda *args, **kwargs: Response())
    case = EvalCase(test_id="same-model", question="q", expected_behavior="answer", expected_facts=["The answer is supported."], category="direct", answerable=True, refusal_expected=False)
    result = evaluate(case, {"answer": "The answer is supported.", "sources": [], "evidence": []},
        OllamaSemanticJudge("http://local", "shared", generation_model="shared"))
    assert result.evaluator_metadata["same_model_judging"] is True
    assert result.evaluator_metadata["independent_evaluator"] is False


def test_independent_evaluator_configuration_selects_configured_model(tmp_path, monkeypatch):
    import forgeai.evaluation.optimizer as module
    from forgeai.models import RequirementDocument
    (tmp_path / "doc.md").write_text("The policy requires staff to submit a completed safety report within five business days.", encoding="utf-8")
    req = RequirementDocument(agent_id="judge_test", agent_name="Judge test", purpose="answer", audience="staff", scope="documents", supported_topics=[], unsupported_topics=[], tone="plain", must_answer=[], must_refuse=[], output_format="answer", available_documents=["doc.md"], required_tools=[])
    definition = generate(req, str(tmp_path))
    observed = {}
    class ConfiguredJudge:
        def __init__(self, base_url, model, timeout, generation_model=None):
            observed.update(base_url=base_url, model=model, generation_model=generation_model)
        def __call__(self, case, output):
            return {"scores": {"semantic_correctness": .8, "groundedness": .8, "citation_correctness": .8,
                "completeness": .8, "refusal_correctness": .8, "writing_quality": .8,
                "correctness": .8, "evidence_quality": .8, "refusal_appropriateness": .8, "instruction_compliance": .8},
                "notes": [], "same_model_judging": False, "evaluator_model": "independent-reviewer",
                "independent_evaluator": True, "order_swapped_result": {"decision_changed": False}}
    monkeypatch.setenv("FORGEAI_JUDGE_PROVIDER", "ollama")
    monkeypatch.setenv("FORGEAI_JUDGE_MODEL", "independent-reviewer")
    monkeypatch.setattr(module, "OllamaSemanticJudge", ConfiguredJudge)
    result = module.run_suite(AgentRunner(definition, tmp_path), [EvalCase(test_id="configured", question="What does the policy require?", expected_behavior="answer", category="direct", answerable=True, refusal_expected=False)])[0]
    assert observed["model"] == "independent-reviewer"
    assert observed["generation_model"] is None
    assert result.evaluator_metadata["independent_evaluator"] is True


def test_writing_quality_does_not_compensate_for_wrong_facts():
    case = EvalCase(test_id="fact", question="What is the leave amount?", expected_behavior="answer",
        expected_facts=["Annual leave provides eighteen days"], category="direct", answerable=True, refusal_expected=False)
    long_polished = "A clear and well-organized response is provided. " * 12
    judged = evaluate(case, {"answer": long_polished, "sources": [{"source": "policy.md", "chunk_id": "x"}], "evidence": []})
    assert judged.writing_quality <= .5
    assert judged.correctness < .6 and not judged.passed
    assert judged.deterministic_checks["expected_facts"][0] == judged.correctness


def test_citation_validation_checks_csv_row_and_chunk():
    case = EvalCase(test_id="row-cite", question="Which price?", expected_behavior="answer",
        expected_facts=["unit_price: 84.00"], supporting_source="catalogue.csv", source_filename="catalogue.csv",
        chunk_id="row-2", row_number=2, supporting_quotation="unit_price: 84.00",
        expected_evidence=["unit_price: 84.00"], evidence_validation_status="validated", category="CSV lookup",
        answerable=True, refusal_expected=False)
    wrong_row = evaluate(case, {"answer": "unit_price: 84.00", "sources": [{"source": "catalogue.csv", "chunk_id": "row-2", "row": 3}], "evidence": []})
    right_row = evaluate(case, {"answer": "unit_price: 84.00", "sources": [{"source": "catalogue.csv", "chunk_id": "row-2", "row": 2}], "evidence": []})
    assert wrong_row.citation_correctness == 0 and right_row.citation_correctness == 1


def test_judge_fallback_marks_semantic_unavailable():
    from forgeai.evaluation.judge import score
    case = EvalCase(test_id="fallback", question="q", expected_behavior="refuse", category="unsupported", answerable=False, refusal_expected=True,
        answerability_classification="unsupported_refusal")
    result = evaluate(case, {"answer": "I cannot verify that from the supplied documents.", "sources": [], "evidence": []})
    assert score([result])["semantic_judging_status"].startswith("unavailable")
    assert result.llm_judge_result is None


def test_optimizer_holdout_is_fixed_and_not_selected_against():
    from forgeai.evaluation.optimizer import _holdout_split
    cases = [EvalCase(test_id=f"f{i:02}", question=f"Question {i}?", expected_behavior="answer",
        expected_facts=[f"Fact {i}"], category="direct", answerable=True, refusal_expected=False) for i in range(30)]
    optimization, holdout, limitation = _holdout_split(cases)
    assert len(optimization) == 24 and len(holdout) == 6 and limitation is None
    assert not ({case.test_id for case in optimization} & {case.test_id for case in holdout})
    assert {case.test_id for case in optimization + holdout} == {case.test_id for case in cases}
    small_optimization, small_holdout, limitation = _holdout_split(cases[:20])
    assert len(small_optimization) == 20 and not small_holdout and limitation

from agent_orchestrator.routing import parse_routing_decision


def test_parse_valid_routing_json_after_prose():
    decision = parse_routing_decision(
        'I completed the analysis. {"next":"Researcher","reason":"Need evidence"}',
        {"planner", "researcher", "executor", "critic"},
    )
    assert decision is not None
    assert decision.next_agent == "researcher"
    assert decision.reason == "Need evidence"


def test_reject_unknown_agent():
    assert parse_routing_decision('{"next":"browser","reason":"unsafe"}', {"planner"}) is None


def test_accept_finish_case_insensitively():
    decision = parse_routing_decision('{"next":"FINISH"}', {"planner"})
    assert decision is not None
    assert decision.next_agent == "finish"


def test_malformed_json_returns_none():
    assert parse_routing_decision('{"next": planner}', {"planner"}) is None

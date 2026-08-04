from complexity_card_corpus.domain_taxonomy import domain_group


def test_domain_groups_are_semantic_and_stable() -> None:
    assert domain_group("database_query") == "technology_data"
    assert domain_group("science_passage") == "education_research"
    assert domain_group("policy_memo") == "governance_legal"
    assert domain_group("personal_finance") == "finance_commerce"
    assert domain_group("travel_plan") == "travel_logistics"
    assert domain_group("unknown_topic") == "general_cross_domain"

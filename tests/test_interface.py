from src.app import parse_query_locally, respond, validate_features


def test_parser_extracts_supported_values():
    values = parse_query_locally("I am a 24 year old female swimmer, 172 cm and 63 kg, competing in 2016 Summer.")
    assert values == {"age": 24, "height": 172.0, "weight": 63.0, "year": 2016, "sex": "F", "season": "Summer", "sport": "Swimming"}
def test_incomplete_query_requests_clarification():
    assert "I need" in respond("Will I win a medal?")
def test_validation_identifies_missing_fields():
    valid, missing = validate_features({"age": 20}); assert not valid and "sport" in missing

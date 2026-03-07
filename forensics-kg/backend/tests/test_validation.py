"""Tests for input validation on request models."""
import pytest
from pydantic import ValidationError
from app.models.requests import (
    TextExtractionRequest,
    NaturalLanguageQueryRequest,
    CypherQueryRequest,
    SyntheticFIRRequest,
    SourceType,
)


class TestTextExtractionRequest:
    def test_valid_request(self):
        req = TextExtractionRequest(text="A" * 20, source_type="fir")
        assert req.source_type == SourceType.fir

    def test_text_too_short(self):
        with pytest.raises(ValidationError):
            TextExtractionRequest(text="short")

    def test_text_too_long(self):
        with pytest.raises(ValidationError):
            TextExtractionRequest(text="A" * 50001)

    def test_invalid_source_type(self):
        with pytest.raises(ValidationError):
            TextExtractionRequest(text="A" * 20, source_type="invalid_type")

    def test_valid_source_types(self):
        for st in ["fir", "report", "evidence", "witness", "generic"]:
            req = TextExtractionRequest(text="A" * 20, source_type=st)
            assert req.source_type.value == st


class TestNaturalLanguageQueryRequest:
    def test_valid_request(self):
        req = NaturalLanguageQueryRequest(question="What cases exist?")
        assert req.max_results == 10

    def test_question_too_short(self):
        with pytest.raises(ValidationError):
            NaturalLanguageQueryRequest(question="ab")

    def test_max_results_bounds(self):
        with pytest.raises(ValidationError):
            NaturalLanguageQueryRequest(question="test query", max_results=0)
        with pytest.raises(ValidationError):
            NaturalLanguageQueryRequest(question="test query", max_results=101)


class TestCypherQueryRequest:
    def test_valid_request(self):
        req = CypherQueryRequest(query="MATCH (n) RETURN n")
        assert req.parameters == {}

    def test_query_too_short(self):
        with pytest.raises(ValidationError):
            CypherQueryRequest(query="ab")


class TestSyntheticFIRRequest:
    def test_defaults(self):
        req = SyntheticFIRRequest()
        assert req.count == 10
        assert req.include_bloodstain is True

    def test_count_bounds(self):
        with pytest.raises(ValidationError):
            SyntheticFIRRequest(count=0)
        with pytest.raises(ValidationError):
            SyntheticFIRRequest(count=101)

    def test_too_many_crime_types(self):
        with pytest.raises(ValidationError):
            SyntheticFIRRequest(crime_types=["type"] * 21)

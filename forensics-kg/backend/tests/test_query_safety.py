"""Tests for query safety checks."""
import pytest
from app.services.query.query_service import _is_destructive


class TestDestructiveQueryDetection:
    def test_detach_delete(self):
        assert _is_destructive("MATCH (n) DETACH DELETE n") is True

    def test_delete(self):
        assert _is_destructive("MATCH (n) DELETE n") is True

    def test_drop_constraint(self):
        assert _is_destructive("DROP CONSTRAINT foo") is True

    def test_safe_match_return(self):
        assert _is_destructive("MATCH (n) RETURN n") is False

    def test_safe_set(self):
        assert _is_destructive("MATCH (n) SET n.name = 'foo' RETURN n") is False

    def test_delete_in_string_literal(self):
        # DELETE inside a string should NOT trigger
        assert _is_destructive("MATCH (n) WHERE n.name = 'DELETE' RETURN n") is False

    def test_drop_in_string_literal(self):
        assert _is_destructive('MATCH (n) WHERE n.action = "DROP" RETURN n') is False

    def test_case_insensitive(self):
        assert _is_destructive("match (n) delete n") is True
        assert _is_destructive("MATCH (n) detach DELETE n") is True

    def test_create_is_safe(self):
        assert _is_destructive("CREATE (n:Test {name: 'foo'})") is False

    def test_merge_is_safe(self):
        assert _is_destructive("MERGE (n:Test {id: 1}) RETURN n") is False

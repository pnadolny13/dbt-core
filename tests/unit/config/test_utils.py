import pytest

from dbt.config.utils import normalize_warn_error_options
from dbt.exceptions import DbtExclusivePropertyUseError


class TestNormalizeWarnErrorOptions:
    def test_primary_set(self):
        test_dict = {
            "error": ["SomeWarning"],
        }
        normalize_warn_error_options(test_dict)
        assert len(test_dict) == 1
        assert test_dict["include"] == ["SomeWarning"]

    def test_convert(self):
        test_dict = {"warn": None, "silence": None, "include": ["SomeWarning"]}
        normalize_warn_error_options(test_dict)
        assert test_dict["exclude"] == []
        assert test_dict["include"] == ["SomeWarning"]
        assert test_dict["silence"] == []

    def test_both_keys_set(self):
        test_dict = {
            "warn": ["SomeWarning"],
            "exclude": ["SomeWarning"],
        }
        with pytest.raises(DbtExclusivePropertyUseError):
            normalize_warn_error_options(test_dict)

    def test_empty_dict(self):
        test_dict = {}
        normalize_warn_error_options(test_dict)
        assert test_dict.get("include") is None
        assert test_dict.get("exclude") is None

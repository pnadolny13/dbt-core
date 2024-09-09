import os
import re
from argparse import Namespace
from typing import Any, Dict, Mapping, Set
from unittest import mock

import pytest
import pytz

import dbt_common.exceptions
from dbt.adapters import factory, postgres
from dbt.clients.jinja import MacroStack
from dbt.config.project import VarProvider
from dbt.context import base, docs, macros, providers, query_header
from dbt.context.base import Var
from dbt.contracts.files import FileHash
from dbt.contracts.graph.nodes import (
    DependsOn,
    Macro,
    ModelNode,
    NodeConfig,
    UnitTestNode,
    UnitTestOverrides,
)
from dbt.node_types import NodeType
from dbt_common.events.functions import reset_metadata_vars
from dbt_common.helper_types import WarnErrorOptions
from tests.unit.mock_adapter import adapter_factory
from tests.unit.utils import clear_plugin, config_from_parts_or_dicts, inject_adapter


class TestVar:
    @pytest.fixture
    def model(self):
        return ModelNode(
            alias="model_one",
            name="model_one",
            database="dbt",
            schema="analytics",
            resource_type=NodeType.Model,
            unique_id="model.root.model_one",
            fqn=["root", "model_one"],
            package_name="root",
            original_file_path="model_one.sql",
            refs=[],
            sources=[],
            depends_on=DependsOn(),
            config=NodeConfig.from_dict(
                {
                    "enabled": True,
                    "materialized": "view",
                    "persist_docs": {},
                    "post-hook": [],
                    "pre-hook": [],
                    "vars": {},
                    "quoting": {},
                    "column_types": {},
                    "tags": [],
                }
            ),
            tags=[],
            path="model_one.sql",
            language="sql",
            raw_code="",
            description="",
            columns={},
            checksum=FileHash.from_contents(""),
        )

    @pytest.fixture
    def context(self):
        return mock.MagicMock()

    @pytest.fixture
    def provider(self):
        return VarProvider({})

    @pytest.fixture
    def config(self, provider):
        return mock.MagicMock(config_version=2, vars=provider, cli_vars={}, project_name="root")

    def test_var_default_something(self, model, config, context):
        config.cli_vars = {"foo": "baz"}
        var = providers.RuntimeVar(context, config, model)

        assert var("foo") == "baz"
        assert var("foo", "bar") == "baz"

    def test_var_default_none(self, model, config, context):
        config.cli_vars = {"foo": None}
        var = providers.RuntimeVar(context, config, model)

        assert var("foo") is None
        assert var("foo", "bar") is None

    def test_var_not_defined(self, model, config, context):
        var = providers.RuntimeVar(self.context, config, model)

        assert var("foo", "bar") == "bar"
        with pytest.raises(dbt_common.exceptions.CompilationError):
            var("foo")

    def test_parser_var_default_something(self, model, config, context):
        config.cli_vars = {"foo": "baz"}
        var = providers.ParseVar(context, config, model)
        assert var("foo") == "baz"
        assert var("foo", "bar") == "baz"

    def test_parser_var_default_none(self, model, config, context):
        config.cli_vars = {"foo": None}
        var = providers.ParseVar(context, config, model)
        assert var("foo") is None
        assert var("foo", "bar") is None

    def test_parser_var_not_defined(self, model, config, context):
        # at parse-time, we should not raise if we encounter a missing var
        # that way disabled models don't get parse errors
        var = providers.ParseVar(context, config, model)

        assert var("foo", "bar") == "bar"
        assert var("foo") is None


class TestParseWrapper:
    @pytest.fixture
    def mock_adapter(self):
        mock_config = mock.MagicMock()
        mock_mp_context = mock.MagicMock()
        adapter_class = adapter_factory()
        return adapter_class(mock_config, mock_mp_context)

    @pytest.fixture
    def wrapper(self, mock_adapter):
        namespace = mock.MagicMock()
        return providers.ParseDatabaseWrapper(mock_adapter, namespace)

    @pytest.fixture
    def responder(self, mock_adapter):
        return mock_adapter.responder

    def test_unwrapped_method(self, wrapper, responder):
        assert wrapper.quote("test_value") == '"test_value"'
        responder.quote.assert_called_once_with("test_value")

    def test_wrapped_method(self, wrapper, responder):
        found = wrapper.get_relation("database", "schema", "identifier")
        assert found is None
        responder.get_relation.assert_not_called()


class TestRuntimeWrapper:
    @pytest.fixture
    def mock_adapter(self):
        mock_config = mock.MagicMock()
        mock_config.quoting = {
            "database": True,
            "schema": True,
            "identifier": True,
        }
        mock_mp_context = mock.MagicMock()
        adapter_class = adapter_factory()
        return adapter_class(mock_config, mock_mp_context)

    @pytest.fixture
    def wrapper(self, mock_adapter):
        namespace = mock.MagicMock()
        return providers.RuntimeDatabaseWrapper(mock_adapter, namespace)

    @pytest.fixture
    def responder(self, mock_adapter):
        return mock_adapter.responder

    def test_unwrapped_method(self, wrapper, responder):
        # the 'quote' method isn't wrapped, we should get our expected inputs
        assert wrapper.quote("test_value") == '"test_value"'
        responder.quote.assert_called_once_with("test_value")


def assert_has_keys(required_keys: Set[str], maybe_keys: Set[str], ctx: Dict[str, Any]):
    keys = set(ctx)
    for key in required_keys:
        assert key in keys, f"{key} in required keys but not in context"
        keys.remove(key)
    extras = keys.difference(maybe_keys)
    assert not extras, f"got extra keys in context: {extras}"


REQUIRED_BASE_KEYS = frozenset(
    {
        "context",
        "builtins",
        "dbt_version",
        "var",
        "env_var",
        "return",
        "fromjson",
        "tojson",
        "fromyaml",
        "toyaml",
        "set",
        "set_strict",
        "zip",
        "zip_strict",
        "log",
        "run_started_at",
        "invocation_id",
        "thread_id",
        "modules",
        "flags",
        "print",
        "diff_of_two_dicts",
        "local_md5",
    }
)

REQUIRED_TARGET_KEYS = REQUIRED_BASE_KEYS | {"target"}
REQUIRED_DOCS_KEYS = REQUIRED_TARGET_KEYS | {"project_name"} | {"doc"}
MACROS = frozenset({"macro_a", "macro_b", "root", "dbt"})
REQUIRED_QUERY_HEADER_KEYS = (
    REQUIRED_TARGET_KEYS | {"project_name", "context_macro_stack"} | MACROS
)
REQUIRED_MACRO_KEYS = REQUIRED_QUERY_HEADER_KEYS | {
    "_sql_results",
    "load_result",
    "store_result",
    "store_raw_result",
    "validation",
    "write",
    "render",
    "try_or_compiler_error",
    "load_agate_table",
    "ref",
    "source",
    "metric",
    "config",
    "execute",
    "exceptions",
    "database",
    "schema",
    "adapter",
    "api",
    "column",
    "env",
    "graph",
    "model",
    "pre_hooks",
    "post_hooks",
    "sql",
    "sql_now",
    "adapter_macro",
    "selected_resources",
    "invocation_args_dict",
    "submit_python_job",
    "dbt_metadata_envs",
}
REQUIRED_MODEL_KEYS = REQUIRED_MACRO_KEYS | {"this", "compiled_code"}
MAYBE_KEYS = frozenset({"debug", "defer_relation"})


POSTGRES_PROFILE_DATA = {
    "target": "test",
    "quoting": {},
    "outputs": {
        "test": {
            "type": "postgres",
            "host": "localhost",
            "schema": "analytics",
            "user": "test",
            "pass": "test",
            "dbname": "test",
            "port": 1,
        }
    },
}

PROJECT_DATA = {
    "name": "root",
    "version": "0.1",
    "profile": "test",
    "project-root": os.getcwd(),
    "config-version": 2,
}

PYTZ_COUNTRY_TIMEZONES_LIST = country_timezones_list = [
    (
        ["modules", "pytz", "country_timezones", country_code],
        str(pytz.country_timezones[country_code]),
    )
    for country_code in pytz.country_timezones
]

EXPECTED_MODEL_CONTEXT = {
    ("var",): {},
    (
        "env_var",
    ): "<bound method ProviderContext.env_var of <dbt.context.providers.ModelContext object>>",
    ("return",): "<function BaseContext._return>",
    ("fromjson",): "<function BaseContext.fromjson>",
    ("tojson",): "<function BaseContext.tojson>",
    ("fromyaml",): "<function BaseContext.fromyaml>",
    ("toyaml",): "<function BaseContext.toyaml>",
    ("set",): "<function BaseContext._set>",
    ("set_strict",): "<function BaseContext.set_strict>",
    ("zip",): "<function BaseContext._zip>",
    ("zip_strict",): "<function BaseContext.zip_strict>",
    ("log",): "<function BaseContext.log>",
    ("run_started_at",): "None",
    ("thread_id",): "MainThread",
    ("flags",): {
        "CACHE_SELECTED_ONLY": False,
        "LOG_CACHE_EVENTS": False,
        "FAIL_FAST": False,
        "SEND_ANONYMOUS_USAGE_STATS": True,
        "LOG_PATH": "logs",
        "DEBUG": False,
        "WARN_ERROR": None,
        "INTROSPECT": True,
        "WARN_ERROR_OPTIONS": WarnErrorOptions(include=[], exclude=[]),
        "QUIET": False,
        "PARTIAL_PARSE": True,
        "WRITE_JSON": True,
        "STATIC_PARSER": True,
        "USE_EXPERIMENTAL_PARSER": False,
        "INVOCATION_COMMAND": "dbt unit/context/test_context.py::test_model_runtime_context",
        "PRINTER_WIDTH": 80,
        "VERSION_CHECK": True,
        "LOG_FORMAT": "default",
        "NO_PRINT": None,
        "PROFILES_DIR": None,
        "TARGET_PATH": None,
        "EMPTY": None,
        "INDIRECT_SELECTION": "eager",
        "USE_COLORS": True,
        "FULL_REFRESH": False,
        "STORE_FAILURES": False,
        "WHICH": "run",
    },
    ("print",): "<function BaseContext.print>",
    ("diff_of_two_dicts",): "<function BaseContext.diff_of_two_dicts>",
    ("local_md5",): "<function BaseContext.local_md5>",
    ("project_name",): "root",
    ("context_macro_stack",): "<dbt.clients.jinja.MacroStack object>",
    (
        "load_result",
    ): "<bound method ProviderContext.load_result of <dbt.context.providers.ModelContext object>>",
    (
        "store_result",
    ): "<bound method ProviderContext.store_result of <dbt.context.providers.ModelContext object>>",
    (
        "store_raw_result",
    ): "<bound method ProviderContext.store_raw_result of <dbt.context.providers.ModelContext object>>",
    (
        "write",
    ): "<bound method ProviderContext.write of <dbt.context.providers.ModelContext object>>",
    (
        "render",
    ): "<bound method ProviderContext.render of <dbt.context.providers.ModelContext object>>",
    (
        "try_or_compiler_error",
    ): "<bound method ProviderContext.try_or_compiler_error of <dbt.context.providers.ModelContext object>>",
    (
        "load_agate_table",
    ): "<bound method ProviderContext.load_agate_table of <dbt.context.providers.ModelContext object>>",
    ("ref",): "<dbt.context.providers.RuntimeRefResolver object>",
    ("source",): "<dbt.context.providers.RuntimeSourceResolver object>",
    ("metric",): "<dbt.context.providers.RuntimeMetricResolver object>",
    ("config",): "<dbt.context.providers.RuntimeConfigObject object>",
    ("execute",): "True",
    ("database",): "dbt",
    ("schema",): "analytics",
    ("adapter",): "<dbt.context.providers.RuntimeDatabaseWrapper object>",
    ("column",): "<MagicMock name='get_adapter().Column'>",
    ("graph",): "<MagicMock name='mock.flat_graph'>",
    ("model",): "<MagicMock name='model_one.to_dict()'>",
    ("pre_hooks",): "[]",
    ("post_hooks",): "[]",
    ("sql",): "<MagicMock name='model_one.compiled_code'>",
    ("sql_now",): "<MagicMock name='get_adapter().date_function()'>",
    (
        "adapter_macro",
    ): "<bound method ProviderContext.adapter_macro of <dbt.context.providers.ModelContext object>>",
    ("selected_resources",): "[]",
    (
        "submit_python_job",
    ): "<bound method ProviderContext.submit_python_job of <dbt.context.providers.ModelContext object>>",
    ("compiled_code",): "<MagicMock name='model_one.compiled_code'>",
    ("this",): "<MagicMock name='get_adapter().Relation.create_from()'>",
    ("defer_relation",): "<MagicMock name='get_adapter().Relation.create_from()'>",
    ("macro_a",): "<dbt.clients.jinja.MacroGenerator object>",
    ("macro_b",): "<dbt.clients.jinja.MacroGenerator object>",
    ("builtins", "var"): {},
    (
        "builtins",
        "env_var",
    ): "<bound method ProviderContext.env_var of <dbt.context.providers.ModelContext object>>",
    ("builtins", "return"): "<function BaseContext._return>",
    ("builtins", "fromjson"): "<function BaseContext.fromjson>",
    ("builtins", "tojson"): "<function BaseContext.tojson>",
    ("builtins", "fromyaml"): "<function BaseContext.fromyaml>",
    ("builtins", "toyaml"): "<function BaseContext.toyaml>",
    ("builtins", "set"): "<function BaseContext._set>",
    ("builtins", "set_strict"): "<function BaseContext.set_strict>",
    ("builtins", "zip"): "<function BaseContext._zip>",
    ("builtins", "zip_strict"): "<function BaseContext.zip_strict>",
    ("builtins", "log"): "<function BaseContext.log>",
    ("builtins", "run_started_at"): "None",
    ("builtins", "thread_id"): "MainThread",
    ("builtins", "flags"): {
        "CACHE_SELECTED_ONLY": False,
        "LOG_CACHE_EVENTS": False,
        "FAIL_FAST": False,
        "SEND_ANONYMOUS_USAGE_STATS": True,
        "LOG_PATH": "logs",
        "DEBUG": False,
        "WARN_ERROR": None,
        "INTROSPECT": True,
        "WARN_ERROR_OPTIONS": WarnErrorOptions(include=[], exclude=[]),
        "QUIET": False,
        "PARTIAL_PARSE": True,
        "WRITE_JSON": True,
        "STATIC_PARSER": True,
        "USE_EXPERIMENTAL_PARSER": False,
        "INVOCATION_COMMAND": "dbt unit/context/test_context.py::test_model_runtime_context",
        "PRINTER_WIDTH": 80,
        "VERSION_CHECK": True,
        "LOG_FORMAT": "default",
        "NO_PRINT": None,
        "PROFILES_DIR": None,
        "TARGET_PATH": None,
        "EMPTY": None,
        "INDIRECT_SELECTION": "eager",
        "USE_COLORS": True,
        "FULL_REFRESH": False,
        "STORE_FAILURES": False,
        "WHICH": "run",
    },
    ("builtins", "print"): "<function BaseContext.print>",
    ("builtins", "diff_of_two_dicts"): "<function BaseContext.diff_of_two_dicts>",
    ("builtins", "local_md5"): "<function BaseContext.local_md5>",
    ("builtins", "project_name"): "root",
    ("builtins", "context_macro_stack"): "<dbt.clients.jinja.MacroStack object>",
    (
        "builtins",
        "load_result",
    ): "<bound method ProviderContext.load_result of <dbt.context.providers.ModelContext object>>",
    (
        "builtins",
        "store_result",
    ): "<bound method ProviderContext.store_result of <dbt.context.providers.ModelContext object>>",
    (
        "builtins",
        "store_raw_result",
    ): "<bound method ProviderContext.store_raw_result of <dbt.context.providers.ModelContext object>>",
    (
        "builtins",
        "write",
    ): "<bound method ProviderContext.write of <dbt.context.providers.ModelContext object>>",
    (
        "builtins",
        "render",
    ): "<bound method ProviderContext.render of <dbt.context.providers.ModelContext object>>",
    (
        "builtins",
        "try_or_compiler_error",
    ): "<bound method ProviderContext.try_or_compiler_error of <dbt.context.providers.ModelContext object>>",
    (
        "builtins",
        "load_agate_table",
    ): "<bound method ProviderContext.load_agate_table of <dbt.context.providers.ModelContext object>>",
    ("builtins", "ref"): "<dbt.context.providers.RuntimeRefResolver object>",
    ("builtins", "source"): "<dbt.context.providers.RuntimeSourceResolver object>",
    ("builtins", "metric"): "<dbt.context.providers.RuntimeMetricResolver object>",
    ("builtins", "config"): "<dbt.context.providers.RuntimeConfigObject object>",
    ("builtins", "execute"): "True",
    ("builtins", "database"): "dbt",
    ("builtins", "schema"): "analytics",
    ("builtins", "adapter"): "<dbt.context.providers.RuntimeDatabaseWrapper object>",
    ("builtins", "column"): "<MagicMock name='get_adapter().Column'>",
    ("builtins", "graph"): "<MagicMock name='mock.flat_graph'>",
    ("builtins", "model"): "<MagicMock name='model_one.to_dict()'>",
    ("builtins", "pre_hooks"): "[]",
    ("builtins", "post_hooks"): "[]",
    ("builtins", "sql"): "<MagicMock name='model_one.compiled_code'>",
    ("builtins", "sql_now"): "<MagicMock name='get_adapter().date_function()'>",
    (
        "builtins",
        "adapter_macro",
    ): "<bound method ProviderContext.adapter_macro of <dbt.context.providers.ModelContext object>>",
    ("builtins", "selected_resources"): "[]",
    (
        "builtins",
        "submit_python_job",
    ): "<bound method ProviderContext.submit_python_job of <dbt.context.providers.ModelContext object>>",
    ("builtins", "compiled_code"): "<MagicMock name='model_one.compiled_code'>",
    ("builtins", "this"): "<MagicMock name='get_adapter().Relation.create_from()'>",
    ("builtins", "defer_relation"): "<MagicMock name='get_adapter().Relation.create_from()'>",
    ("target", "host"): "localhost",
    ("target", "port"): "1",
    ("target", "user"): "test",
    ("target", "database"): "test",
    ("target", "schema"): "analytics",
    ("target", "connect_timeout"): "10",
    ("target", "role"): "None",
    ("target", "search_path"): "None",
    ("target", "keepalives_idle"): "0",
    ("target", "sslmode"): "None",
    ("target", "sslcert"): "None",
    ("target", "sslkey"): "None",
    ("target", "sslrootcert"): "None",
    ("target", "application_name"): "dbt",
    ("target", "retries"): "1",
    ("target", "dbname"): "test",
    ("target", "type"): "postgres",
    ("target", "threads"): "1",
    ("target", "name"): "test",
    ("target", "target_name"): "test",
    ("target", "profile_name"): "test",
    ("invocation_args_dict", "profile_dir"): "/dev/null",
    ("invocation_args_dict", "cache_selected_only"): "False",
    ("invocation_args_dict", "send_anonymous_usage_stats"): "True",
    ("invocation_args_dict", "log_path"): "logs",
    ("invocation_args_dict", "introspect"): "True",
    ("invocation_args_dict", "quiet"): "False",
    ("invocation_args_dict", "partial_parse"): "True",
    ("invocation_args_dict", "write_json"): "True",
    ("invocation_args_dict", "static_parser"): "True",
    (
        "invocation_args_dict",
        "invocation_command",
    ): "dbt unit/context/test_context.py::test_model_runtime_context",
    ("invocation_args_dict", "printer_width"): "80",
    ("invocation_args_dict", "version_check"): "True",
    ("invocation_args_dict", "log_format"): "default",
    ("invocation_args_dict", "indirect_selection"): "eager",
    ("invocation_args_dict", "use_colors"): "True",
    ("validation", "any"): "<function ProviderContext.validation.<locals>.validate_any>",
    ("exceptions", "warn"): "<function warn>",
    ("exceptions", "missing_config"): "<function missing_config>",
    ("exceptions", "missing_materialization"): "<function missing_materialization>",
    ("exceptions", "missing_relation"): "<function missing_relation>",
    ("exceptions", "raise_ambiguous_alias"): "<function raise_ambiguous_alias>",
    ("exceptions", "raise_ambiguous_catalog_match"): "<function raise_ambiguous_catalog_match>",
    ("exceptions", "raise_cache_inconsistent"): "<function raise_cache_inconsistent>",
    ("exceptions", "raise_dataclass_not_dict"): "<function raise_dataclass_not_dict>",
    ("exceptions", "raise_compiler_error"): "<function raise_compiler_error>",
    ("exceptions", "raise_database_error"): "<function raise_database_error>",
    ("exceptions", "raise_dep_not_found"): "<function raise_dep_not_found>",
    ("exceptions", "raise_dependency_error"): "<function raise_dependency_error>",
    ("exceptions", "raise_duplicate_patch_name"): "<function raise_duplicate_patch_name>",
    ("exceptions", "raise_duplicate_resource_name"): "<function raise_duplicate_resource_name>",
    (
        "exceptions",
        "raise_invalid_property_yml_version",
    ): "<function raise_invalid_property_yml_version>",
    ("exceptions", "raise_not_implemented"): "<function raise_not_implemented>",
    ("exceptions", "relation_wrong_type"): "<function relation_wrong_type>",
    ("exceptions", "raise_contract_error"): "<function raise_contract_error>",
    ("exceptions", "column_type_missing"): "<function column_type_missing>",
    ("exceptions", "raise_fail_fast_error"): "<function raise_fail_fast_error>",
    (
        "exceptions",
        "warn_snapshot_timestamp_data_types",
    ): "<function warn_snapshot_timestamp_data_types>",
    ("api", "Relation"): "<dbt.context.providers.RelationProxy object>",
    ("api", "Column"): "<MagicMock name='get_adapter().Column'>",
    ("root", "macro_a"): "<dbt.clients.jinja.MacroGenerator object>",
    ("root", "macro_b"): "<dbt.clients.jinja.MacroGenerator object>",
    ("modules", "pytz", "timezone"): "<function timezone>",
    ("modules", "pytz", "utc"): "UTC",
    ("modules", "pytz", "AmbiguousTimeError"): "<class 'pytz.exceptions.AmbiguousTimeError'>",
    ("modules", "pytz", "InvalidTimeError"): "<class 'pytz.exceptions.InvalidTimeError'>",
    ("modules", "pytz", "NonExistentTimeError"): "<class 'pytz.exceptions.NonExistentTimeError'>",
    ("modules", "pytz", "UnknownTimeZoneError"): "<class 'pytz.exceptions.UnknownTimeZoneError'>",
    ("modules", "pytz", "all_timezones"): str(pytz.all_timezones),
    ("modules", "pytz", "all_timezones_set"): pytz.all_timezones_set,
    ("modules", "pytz", "common_timezones"): str(pytz.common_timezones),
    ("modules", "pytz", "common_timezones_set"): set(),
    ("modules", "pytz", "BaseTzInfo"): "<class 'pytz.tzinfo.BaseTzInfo'>",
    ("modules", "pytz", "FixedOffset"): "<function FixedOffset>",
    ("modules", "datetime", "date"): "<class 'datetime.date'>",
    ("modules", "datetime", "datetime"): "<class 'datetime.datetime'>",
    ("modules", "datetime", "time"): "<class 'datetime.time'>",
    ("modules", "datetime", "timedelta"): "<class 'datetime.timedelta'>",
    ("modules", "datetime", "tzinfo"): "<class 'datetime.tzinfo'>",
    ("modules", "re", "match"): "<function match>",
    ("modules", "re", "fullmatch"): "<function fullmatch>",
    ("modules", "re", "search"): "<function search>",
    ("modules", "re", "sub"): "<function sub>",
    ("modules", "re", "subn"): "<function subn>",
    ("modules", "re", "split"): "<function split>",
    ("modules", "re", "findall"): "<function findall>",
    ("modules", "re", "finditer"): "<function finditer>",
    ("modules", "re", "compile"): "<function compile>",
    ("modules", "re", "purge"): "<function purge>",
    ("modules", "re", "template"): "<function template>",
    ("modules", "re", "escape"): "<function escape>",
    ("modules", "re", "error"): "<class 're.error'>",
    ("modules", "re", "Pattern"): "<class 're.Pattern'>",
    ("modules", "re", "Match"): "<class 're.Match'>",
    ("modules", "re", "A"): "re.ASCII",
    ("modules", "re", "I"): "re.IGNORECASE",
    ("modules", "re", "L"): "re.LOCALE",
    ("modules", "re", "M"): "re.MULTILINE",
    ("modules", "re", "S"): "re.DOTALL",
    ("modules", "re", "X"): "re.VERBOSE",
    ("modules", "re", "U"): "re.UNICODE",
    ("modules", "re", "ASCII"): "re.ASCII",
    ("modules", "re", "IGNORECASE"): "re.IGNORECASE",
    ("modules", "re", "LOCALE"): "re.LOCALE",
    ("modules", "re", "MULTILINE"): "re.MULTILINE",
    ("modules", "re", "DOTALL"): "re.DOTALL",
    ("modules", "re", "VERBOSE"): "re.VERBOSE",
    ("modules", "re", "UNICODE"): "re.UNICODE",
    ("modules", "re", "NOFLAG"): "re.NOFLAG",
    ("modules", "re", "RegexFlag"): "<flag 'RegexFlag'>",
    ("modules", "itertools", "count"): "<class 'itertools.count'>",
    ("modules", "itertools", "cycle"): "<class 'itertools.cycle'>",
    ("modules", "itertools", "repeat"): "<class 'itertools.repeat'>",
    ("modules", "itertools", "accumulate"): "<class 'itertools.accumulate'>",
    ("modules", "itertools", "chain"): "<class 'itertools.chain'>",
    ("modules", "itertools", "compress"): "<class 'itertools.compress'>",
    ("modules", "itertools", "islice"): "<class 'itertools.islice'>",
    ("modules", "itertools", "starmap"): "<class 'itertools.starmap'>",
    ("modules", "itertools", "tee"): "<built-in function tee>",
    ("modules", "itertools", "zip_longest"): "<class 'itertools.zip_longest'>",
    ("modules", "itertools", "product"): "<class 'itertools.product'>",
    ("modules", "itertools", "permutations"): "<class 'itertools.permutations'>",
    ("modules", "itertools", "combinations"): "<class 'itertools.combinations'>",
    (
        "modules",
        "itertools",
        "combinations_with_replacement",
    ): "<class 'itertools.combinations_with_replacement'>",
    ("invocation_args_dict", "warn_error_options", "include"): "[]",
    ("invocation_args_dict", "warn_error_options", "exclude"): "[]",
    ("modules", "pytz", "country_timezones", "AD"): "['Europe/Andorra']",
    ("modules", "pytz", "country_timezones", "AE"): "['Asia/Dubai']",
    ("modules", "pytz", "country_timezones", "AF"): "['Asia/Kabul']",
    ("modules", "pytz", "country_timezones", "AG"): "['America/Antigua']",
    ("modules", "pytz", "country_timezones", "AI"): "['America/Anguilla']",
    ("modules", "pytz", "country_timezones", "AL"): "['Europe/Tirane']",
    ("modules", "pytz", "country_timezones", "AM"): "['Asia/Yerevan']",
    ("modules", "pytz", "country_timezones", "AO"): "['Africa/Luanda']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "AQ",
    ): "['Antarctica/McMurdo', 'Antarctica/Casey', 'Antarctica/Davis', 'Antarctica/DumontDUrville', 'Antarctica/Mawson', 'Antarctica/Palmer', 'Antarctica/Rothera', 'Antarctica/Syowa', 'Antarctica/Troll', 'Antarctica/Vostok']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "AR",
    ): "['America/Argentina/Buenos_Aires', 'America/Argentina/Cordoba', 'America/Argentina/Salta', 'America/Argentina/Jujuy', 'America/Argentina/Tucuman', 'America/Argentina/Catamarca', 'America/Argentina/La_Rioja', 'America/Argentina/San_Juan', 'America/Argentina/Mendoza', 'America/Argentina/San_Luis', 'America/Argentina/Rio_Gallegos', 'America/Argentina/Ushuaia']",
    ("modules", "pytz", "country_timezones", "AS"): "['Pacific/Pago_Pago']",
    ("modules", "pytz", "country_timezones", "AT"): "['Europe/Vienna']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "AU",
    ): "['Australia/Lord_Howe', 'Antarctica/Macquarie', 'Australia/Hobart', 'Australia/Melbourne', 'Australia/Sydney', 'Australia/Broken_Hill', 'Australia/Brisbane', 'Australia/Lindeman', 'Australia/Adelaide', 'Australia/Darwin', 'Australia/Perth', 'Australia/Eucla']",
    ("modules", "pytz", "country_timezones", "AW"): "['America/Aruba']",
    ("modules", "pytz", "country_timezones", "AX"): "['Europe/Mariehamn']",
    ("modules", "pytz", "country_timezones", "AZ"): "['Asia/Baku']",
    ("modules", "pytz", "country_timezones", "BA"): "['Europe/Sarajevo']",
    ("modules", "pytz", "country_timezones", "BB"): "['America/Barbados']",
    ("modules", "pytz", "country_timezones", "BD"): "['Asia/Dhaka']",
    ("modules", "pytz", "country_timezones", "BE"): "['Europe/Brussels']",
    ("modules", "pytz", "country_timezones", "BF"): "['Africa/Ouagadougou']",
    ("modules", "pytz", "country_timezones", "BG"): "['Europe/Sofia']",
    ("modules", "pytz", "country_timezones", "BH"): "['Asia/Bahrain']",
    ("modules", "pytz", "country_timezones", "BI"): "['Africa/Bujumbura']",
    ("modules", "pytz", "country_timezones", "BJ"): "['Africa/Porto-Novo']",
    ("modules", "pytz", "country_timezones", "BL"): "['America/St_Barthelemy']",
    ("modules", "pytz", "country_timezones", "BM"): "['Atlantic/Bermuda']",
    ("modules", "pytz", "country_timezones", "BN"): "['Asia/Brunei']",
    ("modules", "pytz", "country_timezones", "BO"): "['America/La_Paz']",
    ("modules", "pytz", "country_timezones", "BQ"): "['America/Kralendijk']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "BR",
    ): "['America/Noronha', 'America/Belem', 'America/Fortaleza', 'America/Recife', 'America/Araguaina', 'America/Maceio', 'America/Bahia', 'America/Sao_Paulo', 'America/Campo_Grande', 'America/Cuiaba', 'America/Santarem', 'America/Porto_Velho', 'America/Boa_Vista', 'America/Manaus', 'America/Eirunepe', 'America/Rio_Branco']",
    ("modules", "pytz", "country_timezones", "BS"): "['America/Nassau']",
    ("modules", "pytz", "country_timezones", "BT"): "['Asia/Thimphu']",
    ("modules", "pytz", "country_timezones", "BW"): "['Africa/Gaborone']",
    ("modules", "pytz", "country_timezones", "BY"): "['Europe/Minsk']",
    ("modules", "pytz", "country_timezones", "BZ"): "['America/Belize']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "CA",
    ): "['America/St_Johns', 'America/Halifax', 'America/Glace_Bay', 'America/Moncton', 'America/Goose_Bay', 'America/Blanc-Sablon', 'America/Toronto', 'America/Iqaluit', 'America/Atikokan', 'America/Winnipeg', 'America/Resolute', 'America/Rankin_Inlet', 'America/Regina', 'America/Swift_Current', 'America/Edmonton', 'America/Cambridge_Bay', 'America/Inuvik', 'America/Creston', 'America/Dawson_Creek', 'America/Fort_Nelson', 'America/Whitehorse', 'America/Dawson', 'America/Vancouver']",
    ("modules", "pytz", "country_timezones", "CC"): "['Indian/Cocos']",
    ("modules", "pytz", "country_timezones", "CD"): "['Africa/Kinshasa', 'Africa/Lubumbashi']",
    ("modules", "pytz", "country_timezones", "CF"): "['Africa/Bangui']",
    ("modules", "pytz", "country_timezones", "CG"): "['Africa/Brazzaville']",
    ("modules", "pytz", "country_timezones", "CH"): "['Europe/Zurich']",
    ("modules", "pytz", "country_timezones", "CI"): "['Africa/Abidjan']",
    ("modules", "pytz", "country_timezones", "CK"): "['Pacific/Rarotonga']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "CL",
    ): "['America/Santiago', 'America/Punta_Arenas', 'Pacific/Easter']",
    ("modules", "pytz", "country_timezones", "CM"): "['Africa/Douala']",
    ("modules", "pytz", "country_timezones", "CN"): "['Asia/Shanghai', 'Asia/Urumqi']",
    ("modules", "pytz", "country_timezones", "CO"): "['America/Bogota']",
    ("modules", "pytz", "country_timezones", "CR"): "['America/Costa_Rica']",
    ("modules", "pytz", "country_timezones", "CU"): "['America/Havana']",
    ("modules", "pytz", "country_timezones", "CV"): "['Atlantic/Cape_Verde']",
    ("modules", "pytz", "country_timezones", "CW"): "['America/Curacao']",
    ("modules", "pytz", "country_timezones", "CX"): "['Indian/Christmas']",
    ("modules", "pytz", "country_timezones", "CY"): "['Asia/Nicosia', 'Asia/Famagusta']",
    ("modules", "pytz", "country_timezones", "CZ"): "['Europe/Prague']",
    ("modules", "pytz", "country_timezones", "DE"): "['Europe/Berlin', 'Europe/Busingen']",
    ("modules", "pytz", "country_timezones", "DJ"): "['Africa/Djibouti']",
    ("modules", "pytz", "country_timezones", "DK"): "['Europe/Copenhagen']",
    ("modules", "pytz", "country_timezones", "DM"): "['America/Dominica']",
    ("modules", "pytz", "country_timezones", "DO"): "['America/Santo_Domingo']",
    ("modules", "pytz", "country_timezones", "DZ"): "['Africa/Algiers']",
    ("modules", "pytz", "country_timezones", "EC"): "['America/Guayaquil', 'Pacific/Galapagos']",
    ("modules", "pytz", "country_timezones", "EE"): "['Europe/Tallinn']",
    ("modules", "pytz", "country_timezones", "EG"): "['Africa/Cairo']",
    ("modules", "pytz", "country_timezones", "EH"): "['Africa/El_Aaiun']",
    ("modules", "pytz", "country_timezones", "ER"): "['Africa/Asmara']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "ES",
    ): "['Europe/Madrid', 'Africa/Ceuta', 'Atlantic/Canary']",
    ("modules", "pytz", "country_timezones", "ET"): "['Africa/Addis_Ababa']",
    ("modules", "pytz", "country_timezones", "FI"): "['Europe/Helsinki']",
    ("modules", "pytz", "country_timezones", "FJ"): "['Pacific/Fiji']",
    ("modules", "pytz", "country_timezones", "FK"): "['Atlantic/Stanley']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "FM",
    ): "['Pacific/Chuuk', 'Pacific/Pohnpei', 'Pacific/Kosrae']",
    ("modules", "pytz", "country_timezones", "FO"): "['Atlantic/Faroe']",
    ("modules", "pytz", "country_timezones", "FR"): "['Europe/Paris']",
    ("modules", "pytz", "country_timezones", "GA"): "['Africa/Libreville']",
    ("modules", "pytz", "country_timezones", "GB"): "['Europe/London']",
    ("modules", "pytz", "country_timezones", "GD"): "['America/Grenada']",
    ("modules", "pytz", "country_timezones", "GE"): "['Asia/Tbilisi']",
    ("modules", "pytz", "country_timezones", "GF"): "['America/Cayenne']",
    ("modules", "pytz", "country_timezones", "GG"): "['Europe/Guernsey']",
    ("modules", "pytz", "country_timezones", "GH"): "['Africa/Accra']",
    ("modules", "pytz", "country_timezones", "GI"): "['Europe/Gibraltar']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "GL",
    ): "['America/Nuuk', 'America/Danmarkshavn', 'America/Scoresbysund', 'America/Thule']",
    ("modules", "pytz", "country_timezones", "GM"): "['Africa/Banjul']",
    ("modules", "pytz", "country_timezones", "GN"): "['Africa/Conakry']",
    ("modules", "pytz", "country_timezones", "GP"): "['America/Guadeloupe']",
    ("modules", "pytz", "country_timezones", "GQ"): "['Africa/Malabo']",
    ("modules", "pytz", "country_timezones", "GR"): "['Europe/Athens']",
    ("modules", "pytz", "country_timezones", "GS"): "['Atlantic/South_Georgia']",
    ("modules", "pytz", "country_timezones", "GT"): "['America/Guatemala']",
    ("modules", "pytz", "country_timezones", "GU"): "['Pacific/Guam']",
    ("modules", "pytz", "country_timezones", "GW"): "['Africa/Bissau']",
    ("modules", "pytz", "country_timezones", "GY"): "['America/Guyana']",
    ("modules", "pytz", "country_timezones", "HK"): "['Asia/Hong_Kong']",
    ("modules", "pytz", "country_timezones", "HN"): "['America/Tegucigalpa']",
    ("modules", "pytz", "country_timezones", "HR"): "['Europe/Zagreb']",
    ("modules", "pytz", "country_timezones", "HT"): "['America/Port-au-Prince']",
    ("modules", "pytz", "country_timezones", "HU"): "['Europe/Budapest']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "ID",
    ): "['Asia/Jakarta', 'Asia/Pontianak', 'Asia/Makassar', 'Asia/Jayapura']",
    ("modules", "pytz", "country_timezones", "IE"): "['Europe/Dublin']",
    ("modules", "pytz", "country_timezones", "IL"): "['Asia/Jerusalem']",
    ("modules", "pytz", "country_timezones", "IM"): "['Europe/Isle_of_Man']",
    ("modules", "pytz", "country_timezones", "IN"): "['Asia/Kolkata']",
    ("modules", "pytz", "country_timezones", "IO"): "['Indian/Chagos']",
    ("modules", "pytz", "country_timezones", "IQ"): "['Asia/Baghdad']",
    ("modules", "pytz", "country_timezones", "IR"): "['Asia/Tehran']",
    ("modules", "pytz", "country_timezones", "IS"): "['Atlantic/Reykjavik']",
    ("modules", "pytz", "country_timezones", "IT"): "['Europe/Rome']",
    ("modules", "pytz", "country_timezones", "JE"): "['Europe/Jersey']",
    ("modules", "pytz", "country_timezones", "JM"): "['America/Jamaica']",
    ("modules", "pytz", "country_timezones", "JO"): "['Asia/Amman']",
    ("modules", "pytz", "country_timezones", "JP"): "['Asia/Tokyo']",
    ("modules", "pytz", "country_timezones", "KE"): "['Africa/Nairobi']",
    ("modules", "pytz", "country_timezones", "KG"): "['Asia/Bishkek']",
    ("modules", "pytz", "country_timezones", "KH"): "['Asia/Phnom_Penh']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "KI",
    ): "['Pacific/Tarawa', 'Pacific/Kanton', 'Pacific/Kiritimati']",
    ("modules", "pytz", "country_timezones", "KM"): "['Indian/Comoro']",
    ("modules", "pytz", "country_timezones", "KN"): "['America/St_Kitts']",
    ("modules", "pytz", "country_timezones", "KP"): "['Asia/Pyongyang']",
    ("modules", "pytz", "country_timezones", "KR"): "['Asia/Seoul']",
    ("modules", "pytz", "country_timezones", "KW"): "['Asia/Kuwait']",
    ("modules", "pytz", "country_timezones", "KY"): "['America/Cayman']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "KZ",
    ): "['Asia/Almaty', 'Asia/Qyzylorda', 'Asia/Qostanay', 'Asia/Aqtobe', 'Asia/Aqtau', 'Asia/Atyrau', 'Asia/Oral']",
    ("modules", "pytz", "country_timezones", "LA"): "['Asia/Vientiane']",
    ("modules", "pytz", "country_timezones", "LB"): "['Asia/Beirut']",
    ("modules", "pytz", "country_timezones", "LC"): "['America/St_Lucia']",
    ("modules", "pytz", "country_timezones", "LI"): "['Europe/Vaduz']",
    ("modules", "pytz", "country_timezones", "LK"): "['Asia/Colombo']",
    ("modules", "pytz", "country_timezones", "LR"): "['Africa/Monrovia']",
    ("modules", "pytz", "country_timezones", "LS"): "['Africa/Maseru']",
    ("modules", "pytz", "country_timezones", "LT"): "['Europe/Vilnius']",
    ("modules", "pytz", "country_timezones", "LU"): "['Europe/Luxembourg']",
    ("modules", "pytz", "country_timezones", "LV"): "['Europe/Riga']",
    ("modules", "pytz", "country_timezones", "LY"): "['Africa/Tripoli']",
    ("modules", "pytz", "country_timezones", "MA"): "['Africa/Casablanca']",
    ("modules", "pytz", "country_timezones", "MC"): "['Europe/Monaco']",
    ("modules", "pytz", "country_timezones", "MD"): "['Europe/Chisinau']",
    ("modules", "pytz", "country_timezones", "ME"): "['Europe/Podgorica']",
    ("modules", "pytz", "country_timezones", "MF"): "['America/Marigot']",
    ("modules", "pytz", "country_timezones", "MG"): "['Indian/Antananarivo']",
    ("modules", "pytz", "country_timezones", "MH"): "['Pacific/Majuro', 'Pacific/Kwajalein']",
    ("modules", "pytz", "country_timezones", "MK"): "['Europe/Skopje']",
    ("modules", "pytz", "country_timezones", "ML"): "['Africa/Bamako']",
    ("modules", "pytz", "country_timezones", "MM"): "['Asia/Yangon']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "MN",
    ): "['Asia/Ulaanbaatar', 'Asia/Hovd', 'Asia/Choibalsan']",
    ("modules", "pytz", "country_timezones", "MO"): "['Asia/Macau']",
    ("modules", "pytz", "country_timezones", "MP"): "['Pacific/Saipan']",
    ("modules", "pytz", "country_timezones", "MQ"): "['America/Martinique']",
    ("modules", "pytz", "country_timezones", "MR"): "['Africa/Nouakchott']",
    ("modules", "pytz", "country_timezones", "MS"): "['America/Montserrat']",
    ("modules", "pytz", "country_timezones", "MT"): "['Europe/Malta']",
    ("modules", "pytz", "country_timezones", "MU"): "['Indian/Mauritius']",
    ("modules", "pytz", "country_timezones", "MV"): "['Indian/Maldives']",
    ("modules", "pytz", "country_timezones", "MW"): "['Africa/Blantyre']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "MX",
    ): "['America/Mexico_City', 'America/Cancun', 'America/Merida', 'America/Monterrey', 'America/Matamoros', 'America/Chihuahua', 'America/Ciudad_Juarez', 'America/Ojinaga', 'America/Mazatlan', 'America/Bahia_Banderas', 'America/Hermosillo', 'America/Tijuana']",
    ("modules", "pytz", "country_timezones", "MY"): "['Asia/Kuala_Lumpur', 'Asia/Kuching']",
    ("modules", "pytz", "country_timezones", "MZ"): "['Africa/Maputo']",
    ("modules", "pytz", "country_timezones", "NA"): "['Africa/Windhoek']",
    ("modules", "pytz", "country_timezones", "NC"): "['Pacific/Noumea']",
    ("modules", "pytz", "country_timezones", "NE"): "['Africa/Niamey']",
    ("modules", "pytz", "country_timezones", "NF"): "['Pacific/Norfolk']",
    ("modules", "pytz", "country_timezones", "NG"): "['Africa/Lagos']",
    ("modules", "pytz", "country_timezones", "NI"): "['America/Managua']",
    ("modules", "pytz", "country_timezones", "NL"): "['Europe/Amsterdam']",
    ("modules", "pytz", "country_timezones", "NO"): "['Europe/Oslo']",
    ("modules", "pytz", "country_timezones", "NP"): "['Asia/Kathmandu']",
    ("modules", "pytz", "country_timezones", "NR"): "['Pacific/Nauru']",
    ("modules", "pytz", "country_timezones", "NU"): "['Pacific/Niue']",
    ("modules", "pytz", "country_timezones", "NZ"): "['Pacific/Auckland', 'Pacific/Chatham']",
    ("modules", "pytz", "country_timezones", "OM"): "['Asia/Muscat']",
    ("modules", "pytz", "country_timezones", "PA"): "['America/Panama']",
    ("modules", "pytz", "country_timezones", "PE"): "['America/Lima']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "PF",
    ): "['Pacific/Tahiti', 'Pacific/Marquesas', 'Pacific/Gambier']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "PG",
    ): "['Pacific/Port_Moresby', 'Pacific/Bougainville']",
    ("modules", "pytz", "country_timezones", "PH"): "['Asia/Manila']",
    ("modules", "pytz", "country_timezones", "PK"): "['Asia/Karachi']",
    ("modules", "pytz", "country_timezones", "PL"): "['Europe/Warsaw']",
    ("modules", "pytz", "country_timezones", "PM"): "['America/Miquelon']",
    ("modules", "pytz", "country_timezones", "PN"): "['Pacific/Pitcairn']",
    ("modules", "pytz", "country_timezones", "PR"): "['America/Puerto_Rico']",
    ("modules", "pytz", "country_timezones", "PS"): "['Asia/Gaza', 'Asia/Hebron']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "PT",
    ): "['Europe/Lisbon', 'Atlantic/Madeira', 'Atlantic/Azores']",
    ("modules", "pytz", "country_timezones", "PW"): "['Pacific/Palau']",
    ("modules", "pytz", "country_timezones", "PY"): "['America/Asuncion']",
    ("modules", "pytz", "country_timezones", "QA"): "['Asia/Qatar']",
    ("modules", "pytz", "country_timezones", "RE"): "['Indian/Reunion']",
    ("modules", "pytz", "country_timezones", "RO"): "['Europe/Bucharest']",
    ("modules", "pytz", "country_timezones", "RS"): "['Europe/Belgrade']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "RU",
    ): "['Europe/Kaliningrad', 'Europe/Moscow', 'Europe/Kirov', 'Europe/Volgograd', 'Europe/Astrakhan', 'Europe/Saratov', 'Europe/Ulyanovsk', 'Europe/Samara', 'Asia/Yekaterinburg', 'Asia/Omsk', 'Asia/Novosibirsk', 'Asia/Barnaul', 'Asia/Tomsk', 'Asia/Novokuznetsk', 'Asia/Krasnoyarsk', 'Asia/Irkutsk', 'Asia/Chita', 'Asia/Yakutsk', 'Asia/Khandyga', 'Asia/Vladivostok', 'Asia/Ust-Nera', 'Asia/Magadan', 'Asia/Sakhalin', 'Asia/Srednekolymsk', 'Asia/Kamchatka', 'Asia/Anadyr']",
    ("modules", "pytz", "country_timezones", "UA"): "['Europe/Simferopol', 'Europe/Kyiv']",
    ("modules", "pytz", "country_timezones", "RW"): "['Africa/Kigali']",
    ("modules", "pytz", "country_timezones", "SA"): "['Asia/Riyadh']",
    ("modules", "pytz", "country_timezones", "SB"): "['Pacific/Guadalcanal']",
    ("modules", "pytz", "country_timezones", "SC"): "['Indian/Mahe']",
    ("modules", "pytz", "country_timezones", "SD"): "['Africa/Khartoum']",
    ("modules", "pytz", "country_timezones", "SE"): "['Europe/Stockholm']",
    ("modules", "pytz", "country_timezones", "SG"): "['Asia/Singapore']",
    ("modules", "pytz", "country_timezones", "SH"): "['Atlantic/St_Helena']",
    ("modules", "pytz", "country_timezones", "SI"): "['Europe/Ljubljana']",
    ("modules", "pytz", "country_timezones", "SJ"): "['Arctic/Longyearbyen']",
    ("modules", "pytz", "country_timezones", "SK"): "['Europe/Bratislava']",
    ("modules", "pytz", "country_timezones", "SL"): "['Africa/Freetown']",
    ("modules", "pytz", "country_timezones", "SM"): "['Europe/San_Marino']",
    ("modules", "pytz", "country_timezones", "SN"): "['Africa/Dakar']",
    ("modules", "pytz", "country_timezones", "SO"): "['Africa/Mogadishu']",
    ("modules", "pytz", "country_timezones", "SR"): "['America/Paramaribo']",
    ("modules", "pytz", "country_timezones", "SS"): "['Africa/Juba']",
    ("modules", "pytz", "country_timezones", "ST"): "['Africa/Sao_Tome']",
    ("modules", "pytz", "country_timezones", "SV"): "['America/El_Salvador']",
    ("modules", "pytz", "country_timezones", "SX"): "['America/Lower_Princes']",
    ("modules", "pytz", "country_timezones", "SY"): "['Asia/Damascus']",
    ("modules", "pytz", "country_timezones", "SZ"): "['Africa/Mbabane']",
    ("modules", "pytz", "country_timezones", "TC"): "['America/Grand_Turk']",
    ("modules", "pytz", "country_timezones", "TD"): "['Africa/Ndjamena']",
    ("modules", "pytz", "country_timezones", "TF"): "['Indian/Kerguelen']",
    ("modules", "pytz", "country_timezones", "TG"): "['Africa/Lome']",
    ("modules", "pytz", "country_timezones", "TH"): "['Asia/Bangkok']",
    ("modules", "pytz", "country_timezones", "TJ"): "['Asia/Dushanbe']",
    ("modules", "pytz", "country_timezones", "TK"): "['Pacific/Fakaofo']",
    ("modules", "pytz", "country_timezones", "TL"): "['Asia/Dili']",
    ("modules", "pytz", "country_timezones", "TM"): "['Asia/Ashgabat']",
    ("modules", "pytz", "country_timezones", "TN"): "['Africa/Tunis']",
    ("modules", "pytz", "country_timezones", "TO"): "['Pacific/Tongatapu']",
    ("modules", "pytz", "country_timezones", "TR"): "['Europe/Istanbul']",
    ("modules", "pytz", "country_timezones", "TT"): "['America/Port_of_Spain']",
    ("modules", "pytz", "country_timezones", "TV"): "['Pacific/Funafuti']",
    ("modules", "pytz", "country_timezones", "TW"): "['Asia/Taipei']",
    ("modules", "pytz", "country_timezones", "TZ"): "['Africa/Dar_es_Salaam']",
    ("modules", "pytz", "country_timezones", "UG"): "['Africa/Kampala']",
    ("modules", "pytz", "country_timezones", "UM"): "['Pacific/Midway', 'Pacific/Wake']",
    (
        "modules",
        "pytz",
        "country_timezones",
        "US",
    ): "['America/New_York', 'America/Detroit', 'America/Kentucky/Louisville', 'America/Kentucky/Monticello', 'America/Indiana/Indianapolis', 'America/Indiana/Vincennes', 'America/Indiana/Winamac', 'America/Indiana/Marengo', 'America/Indiana/Petersburg', 'America/Indiana/Vevay', 'America/Chicago', 'America/Indiana/Tell_City', 'America/Indiana/Knox', 'America/Menominee', 'America/North_Dakota/Center', 'America/North_Dakota/New_Salem', 'America/North_Dakota/Beulah', 'America/Denver', 'America/Boise', 'America/Phoenix', 'America/Los_Angeles', 'America/Anchorage', 'America/Juneau', 'America/Sitka', 'America/Metlakatla', 'America/Yakutat', 'America/Nome', 'America/Adak', 'Pacific/Honolulu']",
    ("modules", "pytz", "country_timezones", "UY"): "['America/Montevideo']",
    ("modules", "pytz", "country_timezones", "UZ"): "['Asia/Samarkand', 'Asia/Tashkent']",
    ("modules", "pytz", "country_timezones", "VA"): "['Europe/Vatican']",
    ("modules", "pytz", "country_timezones", "VC"): "['America/St_Vincent']",
    ("modules", "pytz", "country_timezones", "VE"): "['America/Caracas']",
    ("modules", "pytz", "country_timezones", "VG"): "['America/Tortola']",
    ("modules", "pytz", "country_timezones", "VI"): "['America/St_Thomas']",
    ("modules", "pytz", "country_timezones", "VN"): "['Asia/Ho_Chi_Minh']",
    ("modules", "pytz", "country_timezones", "VU"): "['Pacific/Efate']",
    ("modules", "pytz", "country_timezones", "WF"): "['Pacific/Wallis']",
    ("modules", "pytz", "country_timezones", "WS"): "['Pacific/Apia']",
    ("modules", "pytz", "country_timezones", "YE"): "['Asia/Aden']",
    ("modules", "pytz", "country_timezones", "YT"): "['Indian/Mayotte']",
    ("modules", "pytz", "country_timezones", "ZA"): "['Africa/Johannesburg']",
    ("modules", "pytz", "country_timezones", "ZM"): "['Africa/Lusaka']",
    ("modules", "pytz", "country_timezones", "ZW"): "['Africa/Harare']",
    ("modules", "pytz", "country_names", "AD"): "Andorra",
    ("modules", "pytz", "country_names", "AE"): "United Arab Emirates",
    ("modules", "pytz", "country_names", "AF"): "Afghanistan",
    ("modules", "pytz", "country_names", "AG"): "Antigua & Barbuda",
    ("modules", "pytz", "country_names", "AI"): "Anguilla",
    ("modules", "pytz", "country_names", "AL"): "Albania",
    ("modules", "pytz", "country_names", "AM"): "Armenia",
    ("modules", "pytz", "country_names", "AO"): "Angola",
    ("modules", "pytz", "country_names", "AQ"): "Antarctica",
    ("modules", "pytz", "country_names", "AR"): "Argentina",
    ("modules", "pytz", "country_names", "AS"): "Samoa (American)",
    ("modules", "pytz", "country_names", "AT"): "Austria",
    ("modules", "pytz", "country_names", "AU"): "Australia",
    ("modules", "pytz", "country_names", "AW"): "Aruba",
    ("modules", "pytz", "country_names", "AX"): "Åland Islands",
    ("modules", "pytz", "country_names", "AZ"): "Azerbaijan",
    ("modules", "pytz", "country_names", "BA"): "Bosnia & Herzegovina",
    ("modules", "pytz", "country_names", "BB"): "Barbados",
    ("modules", "pytz", "country_names", "BD"): "Bangladesh",
    ("modules", "pytz", "country_names", "BE"): "Belgium",
    ("modules", "pytz", "country_names", "BF"): "Burkina Faso",
    ("modules", "pytz", "country_names", "BG"): "Bulgaria",
    ("modules", "pytz", "country_names", "BH"): "Bahrain",
    ("modules", "pytz", "country_names", "BI"): "Burundi",
    ("modules", "pytz", "country_names", "BJ"): "Benin",
    ("modules", "pytz", "country_names", "BL"): "St Barthelemy",
    ("modules", "pytz", "country_names", "BM"): "Bermuda",
    ("modules", "pytz", "country_names", "BN"): "Brunei",
    ("modules", "pytz", "country_names", "BO"): "Bolivia",
    ("modules", "pytz", "country_names", "BQ"): "Caribbean NL",
    ("modules", "pytz", "country_names", "BR"): "Brazil",
    ("modules", "pytz", "country_names", "BS"): "Bahamas",
    ("modules", "pytz", "country_names", "BT"): "Bhutan",
    ("modules", "pytz", "country_names", "BV"): "Bouvet Island",
    ("modules", "pytz", "country_names", "BW"): "Botswana",
    ("modules", "pytz", "country_names", "BY"): "Belarus",
    ("modules", "pytz", "country_names", "BZ"): "Belize",
    ("modules", "pytz", "country_names", "CA"): "Canada",
    ("modules", "pytz", "country_names", "CC"): "Cocos (Keeling) Islands",
    ("modules", "pytz", "country_names", "CD"): "Congo (Dem. Rep.)",
    ("modules", "pytz", "country_names", "CF"): "Central African Rep.",
    ("modules", "pytz", "country_names", "CG"): "Congo (Rep.)",
    ("modules", "pytz", "country_names", "CH"): "Switzerland",
    ("modules", "pytz", "country_names", "CI"): "Côte d'Ivoire",
    ("modules", "pytz", "country_names", "CK"): "Cook Islands",
    ("modules", "pytz", "country_names", "CL"): "Chile",
    ("modules", "pytz", "country_names", "CM"): "Cameroon",
    ("modules", "pytz", "country_names", "CN"): "China",
    ("modules", "pytz", "country_names", "CO"): "Colombia",
    ("modules", "pytz", "country_names", "CR"): "Costa Rica",
    ("modules", "pytz", "country_names", "CU"): "Cuba",
    ("modules", "pytz", "country_names", "CV"): "Cape Verde",
    ("modules", "pytz", "country_names", "CW"): "Curaçao",
    ("modules", "pytz", "country_names", "CX"): "Christmas Island",
    ("modules", "pytz", "country_names", "CY"): "Cyprus",
    ("modules", "pytz", "country_names", "CZ"): "Czech Republic",
    ("modules", "pytz", "country_names", "DE"): "Germany",
    ("modules", "pytz", "country_names", "DJ"): "Djibouti",
    ("modules", "pytz", "country_names", "DK"): "Denmark",
    ("modules", "pytz", "country_names", "DM"): "Dominica",
    ("modules", "pytz", "country_names", "DO"): "Dominican Republic",
    ("modules", "pytz", "country_names", "DZ"): "Algeria",
    ("modules", "pytz", "country_names", "EC"): "Ecuador",
    ("modules", "pytz", "country_names", "EE"): "Estonia",
    ("modules", "pytz", "country_names", "EG"): "Egypt",
    ("modules", "pytz", "country_names", "EH"): "Western Sahara",
    ("modules", "pytz", "country_names", "ER"): "Eritrea",
    ("modules", "pytz", "country_names", "ES"): "Spain",
    ("modules", "pytz", "country_names", "ET"): "Ethiopia",
    ("modules", "pytz", "country_names", "FI"): "Finland",
    ("modules", "pytz", "country_names", "FJ"): "Fiji",
    ("modules", "pytz", "country_names", "FK"): "Falkland Islands",
    ("modules", "pytz", "country_names", "FM"): "Micronesia",
    ("modules", "pytz", "country_names", "FO"): "Faroe Islands",
    ("modules", "pytz", "country_names", "FR"): "France",
    ("modules", "pytz", "country_names", "GA"): "Gabon",
    ("modules", "pytz", "country_names", "GB"): "Britain (UK)",
    ("modules", "pytz", "country_names", "GD"): "Grenada",
    ("modules", "pytz", "country_names", "GE"): "Georgia",
    ("modules", "pytz", "country_names", "GF"): "French Guiana",
    ("modules", "pytz", "country_names", "GG"): "Guernsey",
    ("modules", "pytz", "country_names", "GH"): "Ghana",
    ("modules", "pytz", "country_names", "GI"): "Gibraltar",
    ("modules", "pytz", "country_names", "GL"): "Greenland",
    ("modules", "pytz", "country_names", "GM"): "Gambia",
    ("modules", "pytz", "country_names", "GN"): "Guinea",
    ("modules", "pytz", "country_names", "GP"): "Guadeloupe",
    ("modules", "pytz", "country_names", "GQ"): "Equatorial Guinea",
    ("modules", "pytz", "country_names", "GR"): "Greece",
    ("modules", "pytz", "country_names", "GS"): "South Georgia & the South Sandwich Islands",
    ("modules", "pytz", "country_names", "GT"): "Guatemala",
    ("modules", "pytz", "country_names", "GU"): "Guam",
    ("modules", "pytz", "country_names", "GW"): "Guinea-Bissau",
    ("modules", "pytz", "country_names", "GY"): "Guyana",
    ("modules", "pytz", "country_names", "HK"): "Hong Kong",
    ("modules", "pytz", "country_names", "HM"): "Heard Island & McDonald Islands",
    ("modules", "pytz", "country_names", "HN"): "Honduras",
    ("modules", "pytz", "country_names", "HR"): "Croatia",
    ("modules", "pytz", "country_names", "HT"): "Haiti",
    ("modules", "pytz", "country_names", "HU"): "Hungary",
    ("modules", "pytz", "country_names", "ID"): "Indonesia",
    ("modules", "pytz", "country_names", "IE"): "Ireland",
    ("modules", "pytz", "country_names", "IL"): "Israel",
    ("modules", "pytz", "country_names", "IM"): "Isle of Man",
    ("modules", "pytz", "country_names", "IN"): "India",
    ("modules", "pytz", "country_names", "IO"): "British Indian Ocean Territory",
    ("modules", "pytz", "country_names", "IQ"): "Iraq",
    ("modules", "pytz", "country_names", "IR"): "Iran",
    ("modules", "pytz", "country_names", "IS"): "Iceland",
    ("modules", "pytz", "country_names", "IT"): "Italy",
    ("modules", "pytz", "country_names", "JE"): "Jersey",
    ("modules", "pytz", "country_names", "JM"): "Jamaica",
    ("modules", "pytz", "country_names", "JO"): "Jordan",
    ("modules", "pytz", "country_names", "JP"): "Japan",
    ("modules", "pytz", "country_names", "KE"): "Kenya",
    ("modules", "pytz", "country_names", "KG"): "Kyrgyzstan",
    ("modules", "pytz", "country_names", "KH"): "Cambodia",
    ("modules", "pytz", "country_names", "KI"): "Kiribati",
    ("modules", "pytz", "country_names", "KM"): "Comoros",
    ("modules", "pytz", "country_names", "KN"): "St Kitts & Nevis",
    ("modules", "pytz", "country_names", "KP"): "Korea (North)",
    ("modules", "pytz", "country_names", "KR"): "Korea (South)",
    ("modules", "pytz", "country_names", "KW"): "Kuwait",
    ("modules", "pytz", "country_names", "KY"): "Cayman Islands",
    ("modules", "pytz", "country_names", "KZ"): "Kazakhstan",
    ("modules", "pytz", "country_names", "LA"): "Laos",
    ("modules", "pytz", "country_names", "LB"): "Lebanon",
    ("modules", "pytz", "country_names", "LC"): "St Lucia",
    ("modules", "pytz", "country_names", "LI"): "Liechtenstein",
    ("modules", "pytz", "country_names", "LK"): "Sri Lanka",
    ("modules", "pytz", "country_names", "LR"): "Liberia",
    ("modules", "pytz", "country_names", "LS"): "Lesotho",
    ("modules", "pytz", "country_names", "LT"): "Lithuania",
    ("modules", "pytz", "country_names", "LU"): "Luxembourg",
    ("modules", "pytz", "country_names", "LV"): "Latvia",
    ("modules", "pytz", "country_names", "LY"): "Libya",
    ("modules", "pytz", "country_names", "MA"): "Morocco",
    ("modules", "pytz", "country_names", "MC"): "Monaco",
    ("modules", "pytz", "country_names", "MD"): "Moldova",
    ("modules", "pytz", "country_names", "ME"): "Montenegro",
    ("modules", "pytz", "country_names", "MF"): "St Martin (French)",
    ("modules", "pytz", "country_names", "MG"): "Madagascar",
    ("modules", "pytz", "country_names", "MH"): "Marshall Islands",
    ("modules", "pytz", "country_names", "MK"): "North Macedonia",
    ("modules", "pytz", "country_names", "ML"): "Mali",
    ("modules", "pytz", "country_names", "MM"): "Myanmar (Burma)",
    ("modules", "pytz", "country_names", "MN"): "Mongolia",
    ("modules", "pytz", "country_names", "MO"): "Macau",
    ("modules", "pytz", "country_names", "MP"): "Northern Mariana Islands",
    ("modules", "pytz", "country_names", "MQ"): "Martinique",
    ("modules", "pytz", "country_names", "MR"): "Mauritania",
    ("modules", "pytz", "country_names", "MS"): "Montserrat",
    ("modules", "pytz", "country_names", "MT"): "Malta",
    ("modules", "pytz", "country_names", "MU"): "Mauritius",
    ("modules", "pytz", "country_names", "MV"): "Maldives",
    ("modules", "pytz", "country_names", "MW"): "Malawi",
    ("modules", "pytz", "country_names", "MX"): "Mexico",
    ("modules", "pytz", "country_names", "MY"): "Malaysia",
    ("modules", "pytz", "country_names", "MZ"): "Mozambique",
    ("modules", "pytz", "country_names", "NA"): "Namibia",
    ("modules", "pytz", "country_names", "NC"): "New Caledonia",
    ("modules", "pytz", "country_names", "NE"): "Niger",
    ("modules", "pytz", "country_names", "NF"): "Norfolk Island",
    ("modules", "pytz", "country_names", "NG"): "Nigeria",
    ("modules", "pytz", "country_names", "NI"): "Nicaragua",
    ("modules", "pytz", "country_names", "NL"): "Netherlands",
    ("modules", "pytz", "country_names", "NO"): "Norway",
    ("modules", "pytz", "country_names", "NP"): "Nepal",
    ("modules", "pytz", "country_names", "NR"): "Nauru",
    ("modules", "pytz", "country_names", "NU"): "Niue",
    ("modules", "pytz", "country_names", "NZ"): "New Zealand",
    ("modules", "pytz", "country_names", "OM"): "Oman",
    ("modules", "pytz", "country_names", "PA"): "Panama",
    ("modules", "pytz", "country_names", "PE"): "Peru",
    ("modules", "pytz", "country_names", "PF"): "French Polynesia",
    ("modules", "pytz", "country_names", "PG"): "Papua New Guinea",
    ("modules", "pytz", "country_names", "PH"): "Philippines",
    ("modules", "pytz", "country_names", "PK"): "Pakistan",
    ("modules", "pytz", "country_names", "PL"): "Poland",
    ("modules", "pytz", "country_names", "PM"): "St Pierre & Miquelon",
    ("modules", "pytz", "country_names", "PN"): "Pitcairn",
    ("modules", "pytz", "country_names", "PR"): "Puerto Rico",
    ("modules", "pytz", "country_names", "PS"): "Palestine",
    ("modules", "pytz", "country_names", "PT"): "Portugal",
    ("modules", "pytz", "country_names", "PW"): "Palau",
    ("modules", "pytz", "country_names", "PY"): "Paraguay",
    ("modules", "pytz", "country_names", "QA"): "Qatar",
    ("modules", "pytz", "country_names", "RE"): "Réunion",
    ("modules", "pytz", "country_names", "RO"): "Romania",
    ("modules", "pytz", "country_names", "RS"): "Serbia",
    ("modules", "pytz", "country_names", "RU"): "Russia",
    ("modules", "pytz", "country_names", "RW"): "Rwanda",
    ("modules", "pytz", "country_names", "SA"): "Saudi Arabia",
    ("modules", "pytz", "country_names", "SB"): "Solomon Islands",
    ("modules", "pytz", "country_names", "SC"): "Seychelles",
    ("modules", "pytz", "country_names", "SD"): "Sudan",
    ("modules", "pytz", "country_names", "SE"): "Sweden",
    ("modules", "pytz", "country_names", "SG"): "Singapore",
    ("modules", "pytz", "country_names", "SH"): "St Helena",
    ("modules", "pytz", "country_names", "SI"): "Slovenia",
    ("modules", "pytz", "country_names", "SJ"): "Svalbard & Jan Mayen",
    ("modules", "pytz", "country_names", "SK"): "Slovakia",
    ("modules", "pytz", "country_names", "SL"): "Sierra Leone",
    ("modules", "pytz", "country_names", "SM"): "San Marino",
    ("modules", "pytz", "country_names", "SN"): "Senegal",
    ("modules", "pytz", "country_names", "SO"): "Somalia",
    ("modules", "pytz", "country_names", "SR"): "Suriname",
    ("modules", "pytz", "country_names", "SS"): "South Sudan",
    ("modules", "pytz", "country_names", "ST"): "Sao Tome & Principe",
    ("modules", "pytz", "country_names", "SV"): "El Salvador",
    ("modules", "pytz", "country_names", "SX"): "St Maarten (Dutch)",
    ("modules", "pytz", "country_names", "SY"): "Syria",
    ("modules", "pytz", "country_names", "SZ"): "Eswatini (Swaziland)",
    ("modules", "pytz", "country_names", "TC"): "Turks & Caicos Is",
    ("modules", "pytz", "country_names", "TD"): "Chad",
    ("modules", "pytz", "country_names", "TF"): "French S. Terr.",
    ("modules", "pytz", "country_names", "TG"): "Togo",
    ("modules", "pytz", "country_names", "TH"): "Thailand",
    ("modules", "pytz", "country_names", "TJ"): "Tajikistan",
    ("modules", "pytz", "country_names", "TK"): "Tokelau",
    ("modules", "pytz", "country_names", "TL"): "East Timor",
    ("modules", "pytz", "country_names", "TM"): "Turkmenistan",
    ("modules", "pytz", "country_names", "TN"): "Tunisia",
    ("modules", "pytz", "country_names", "TO"): "Tonga",
    ("modules", "pytz", "country_names", "TR"): "Turkey",
    ("modules", "pytz", "country_names", "TT"): "Trinidad & Tobago",
    ("modules", "pytz", "country_names", "TV"): "Tuvalu",
    ("modules", "pytz", "country_names", "TW"): "Taiwan",
    ("modules", "pytz", "country_names", "TZ"): "Tanzania",
    ("modules", "pytz", "country_names", "UA"): "Ukraine",
    ("modules", "pytz", "country_names", "UG"): "Uganda",
    ("modules", "pytz", "country_names", "UM"): "US minor outlying islands",
    ("modules", "pytz", "country_names", "US"): "United States",
    ("modules", "pytz", "country_names", "UY"): "Uruguay",
    ("modules", "pytz", "country_names", "UZ"): "Uzbekistan",
    ("modules", "pytz", "country_names", "VA"): "Vatican City",
    ("modules", "pytz", "country_names", "VC"): "St Vincent",
    ("modules", "pytz", "country_names", "VE"): "Venezuela",
    ("modules", "pytz", "country_names", "VG"): "Virgin Islands (UK)",
    ("modules", "pytz", "country_names", "VI"): "Virgin Islands (US)",
    ("modules", "pytz", "country_names", "VN"): "Vietnam",
    ("modules", "pytz", "country_names", "VU"): "Vanuatu",
    ("modules", "pytz", "country_names", "WF"): "Wallis & Futuna",
    ("modules", "pytz", "country_names", "WS"): "Samoa (western)",
    ("modules", "pytz", "country_names", "YE"): "Yemen",
    ("modules", "pytz", "country_names", "YT"): "Mayotte",
    ("modules", "pytz", "country_names", "ZA"): "South Africa",
    ("modules", "pytz", "country_names", "ZM"): "Zambia",
    ("modules", "pytz", "country_names", "ZW"): "Zimbabwe",
}


def model():
    return ModelNode(
        alias="model_one",
        name="model_one",
        database="dbt",
        schema="analytics",
        resource_type=NodeType.Model,
        unique_id="model.root.model_one",
        fqn=["root", "model_one"],
        package_name="root",
        original_file_path="model_one.sql",
        refs=[],
        sources=[],
        depends_on=DependsOn(),
        config=NodeConfig.from_dict(
            {
                "enabled": True,
                "materialized": "view",
                "persist_docs": {},
                "post-hook": [],
                "pre-hook": [],
                "vars": {},
                "quoting": {},
                "column_types": {},
                "tags": [],
            }
        ),
        tags=[],
        path="model_one.sql",
        language="sql",
        raw_code="",
        description="",
        columns={},
    )


def test_base_context():
    ctx = base.generate_base_context({})
    assert_has_keys(REQUIRED_BASE_KEYS, MAYBE_KEYS, ctx)


def mock_macro(name, package_name):
    macro = mock.MagicMock(
        __class__=Macro,
        package_name=package_name,
        resource_type="macro",
        unique_id=f"macro.{package_name}.{name}",
    )
    # Mock(name=...) does not set the `name` attribute, this does.
    macro.name = name
    return macro


def mock_manifest(config, additional_macros=None):
    default_macro_names = ["macro_a", "macro_b"]
    default_macros = [mock_macro(name, config.project_name) for name in default_macro_names]
    additional_macros = additional_macros or []
    all_macros = default_macros + additional_macros

    manifest_macros = {}
    macros_by_package = {}
    for macro in all_macros:
        manifest_macros[macro.unique_id] = macro
        if macro.package_name not in macros_by_package:
            macros_by_package[macro.package_name] = {}
        macro_package = macros_by_package[macro.package_name]
        macro_package[macro.name] = macro

    def gmbp():
        return macros_by_package

    m = mock.MagicMock(macros=manifest_macros)
    m.get_macros_by_package = gmbp
    return m


def mock_model():
    return mock.MagicMock(
        __class__=ModelNode,
        alias="model_one",
        name="model_one",
        database="dbt",
        schema="analytics",
        resource_type=NodeType.Model,
        unique_id="model.root.model_one",
        fqn=["root", "model_one"],
        package_name="root",
        original_file_path="model_one.sql",
        refs=[],
        sources=[],
        depends_on=DependsOn(),
        config=NodeConfig.from_dict(
            {
                "enabled": True,
                "materialized": "view",
                "persist_docs": {},
                "post-hook": [],
                "pre-hook": [],
                "vars": {},
                "quoting": {},
                "column_types": {},
                "tags": [],
            }
        ),
        tags=[],
        path="model_one.sql",
        language="sql",
        raw_code="",
        description="",
        columns={},
    )


def mock_unit_test_node():
    return mock.MagicMock(
        __class__=UnitTestNode,
        resource_type=NodeType.Unit,
        tested_node_unique_id="model.root.model_one",
    )


@pytest.fixture
def get_adapter():
    with mock.patch.object(providers, "get_adapter") as patch:
        yield patch


@pytest.fixture
def get_include_paths():
    with mock.patch.object(factory, "get_include_paths") as patch:
        patch.return_value = []
        yield patch


@pytest.fixture
def config_postgres():
    return config_from_parts_or_dicts(PROJECT_DATA, POSTGRES_PROFILE_DATA)


@pytest.fixture
def manifest_fx(config_postgres):
    return mock_manifest(config_postgres)


@pytest.fixture
def postgres_adapter(config_postgres, get_adapter):
    adapter = postgres.PostgresAdapter(config_postgres)
    inject_adapter(adapter, postgres.Plugin)
    get_adapter.return_value = adapter
    yield adapter
    clear_plugin(postgres.Plugin)


def test_query_header_context(config_postgres, manifest_fx):
    ctx = query_header.generate_query_header_context(
        config=config_postgres,
        manifest=manifest_fx,
    )
    assert_has_keys(REQUIRED_QUERY_HEADER_KEYS, MAYBE_KEYS, ctx)


def test_macro_runtime_context(config_postgres, manifest_fx, get_adapter, get_include_paths):
    ctx = providers.generate_runtime_macro_context(
        macro=manifest_fx.macros["macro.root.macro_a"],
        config=config_postgres,
        manifest=manifest_fx,
        package_name="root",
    )
    assert_has_keys(REQUIRED_MACRO_KEYS, MAYBE_KEYS, ctx)


def test_invocation_args_to_dict_in_macro_runtime_context(
    config_postgres, manifest_fx, get_adapter, get_include_paths
):
    ctx = providers.generate_runtime_macro_context(
        macro=manifest_fx.macros["macro.root.macro_a"],
        config=config_postgres,
        manifest=manifest_fx,
        package_name="root",
    )

    # Comes from dbt/flags.py as they are the only values set that aren't None at default
    assert ctx["invocation_args_dict"]["printer_width"] == 80

    # Comes from unit/utils.py config_from_parts_or_dicts method
    assert ctx["invocation_args_dict"]["profile_dir"] == "/dev/null"

    assert isinstance(ctx["invocation_args_dict"]["warn_error_options"], Dict)
    assert ctx["invocation_args_dict"]["warn_error_options"] == {"include": [], "exclude": []}


def test_model_parse_context(config_postgres, manifest_fx, get_adapter, get_include_paths):
    ctx = providers.generate_parser_model_context(
        model=mock_model(),
        config=config_postgres,
        manifest=manifest_fx,
        context_config=mock.MagicMock(),
    )
    assert_has_keys(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)


def walk_dict(dictionary):
    skip_paths = [
        ["invocation_id"],
        ["builtins", "invocation_id"],
        ["dbt_version"],
        ["builtins", "dbt_version"],
    ]

    stack = [(dictionary, [])]
    visited = set()  # Set to keep track of visited dictionary objects

    def clean_value(value):
        if isinstance(value, set):
            return set(value)
        elif isinstance(value, Namespace):
            return value.__dict__
        elif isinstance(value, Var):
            return {k: v for k, v in value._merged.items()}
        else:
            value_str = str(value)
            value_str = re.sub(r" at 0x[0-9a-fA-F]+>", ">", value_str)
            value_str = re.sub(r" id='[0-9]+'>", ">", value_str)
            return value_str

    while stack:
        current_dict, path = stack.pop(0)

        if id(current_dict) in visited:
            continue

        visited.add(id(current_dict))

        for key, value in current_dict.items():
            current_path = path + [key]

            if isinstance(value, Mapping):
                stack.append((value, current_path))
            else:
                if current_path not in skip_paths:
                    yield (tuple(current_path), clean_value(value))


def test_model_runtime_context(config_postgres, manifest_fx, get_adapter, get_include_paths):
    ctx = providers.generate_runtime_model_context(
        model=mock_model(),
        config=config_postgres,
        manifest=manifest_fx,
    )
    actual_model_context = {k: v for (k, v) in walk_dict(ctx)}
    assert actual_model_context == EXPECTED_MODEL_CONTEXT


def test_docs_runtime_context(config_postgres):
    ctx = docs.generate_runtime_docs_context(config_postgres, mock_model(), [], "root")
    assert_has_keys(REQUIRED_DOCS_KEYS, MAYBE_KEYS, ctx)


def test_macro_namespace_duplicates(config_postgres, manifest_fx):
    mn = macros.MacroNamespaceBuilder("root", "search", MacroStack(), ["dbt_postgres", "dbt"])
    mn.add_macros(manifest_fx.macros.values(), {})

    # same pkg, same name: error
    with pytest.raises(dbt_common.exceptions.CompilationError):
        mn.add_macro(mock_macro("macro_a", "root"), {})

    # different pkg, same name: no error
    mn.add_macros(mock_macro("macro_a", "dbt"), {})


def test_macro_namespace(config_postgres, manifest_fx):
    mn = macros.MacroNamespaceBuilder("root", "search", MacroStack(), ["dbt_postgres", "dbt"])

    mbp = manifest_fx.get_macros_by_package()
    dbt_macro = mock_macro("some_macro", "dbt")
    mbp["dbt"] = {"some_macro": dbt_macro}

    # same namespace, same name, different pkg!
    pg_macro = mock_macro("some_macro", "dbt_postgres")
    mbp["dbt_postgres"] = {"some_macro": pg_macro}

    # same name, different package
    package_macro = mock_macro("some_macro", "root")
    mbp["root"]["some_macro"] = package_macro

    namespace = mn.build_namespace(mbp, {})
    dct = dict(namespace)
    for result in [dct, namespace]:
        assert "dbt" in result
        assert "root" in result
        assert "some_macro" in result
        assert "dbt_postgres" not in result
        # tests __len__
        assert len(result) == 5
        # tests __iter__
        assert set(result) == {"dbt", "root", "some_macro", "macro_a", "macro_b"}
        assert len(result["dbt"]) == 1
        # from the regular manifest + some_macro
        assert len(result["root"]) == 3
        assert result["dbt"]["some_macro"].macro is pg_macro
        assert result["root"]["some_macro"].macro is package_macro
        assert result["some_macro"].macro is package_macro


def test_dbt_metadata_envs(
    monkeypatch, config_postgres, manifest_fx, get_adapter, get_include_paths
):
    reset_metadata_vars()

    envs = {
        "DBT_ENV_CUSTOM_ENV_RUN_ID": 1234,
        "DBT_ENV_CUSTOM_ENV_JOB_ID": 5678,
        "DBT_ENV_RUN_ID": 91011,
        "RANDOM_ENV": 121314,
    }
    monkeypatch.setattr(os, "environ", envs)

    ctx = providers.generate_runtime_macro_context(
        macro=manifest_fx.macros["macro.root.macro_a"],
        config=config_postgres,
        manifest=manifest_fx,
        package_name="root",
    )

    assert ctx["dbt_metadata_envs"] == {"JOB_ID": 5678, "RUN_ID": 1234}

    # cleanup
    reset_metadata_vars()


def test_unit_test_runtime_context(config_postgres, manifest_fx, get_adapter, get_include_paths):
    ctx = providers.generate_runtime_unit_test_context(
        unit_test=mock_unit_test_node(),
        config=config_postgres,
        manifest=manifest_fx,
    )
    assert_has_keys(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)


def test_unit_test_runtime_context_macro_overrides_global(
    config_postgres, manifest_fx, get_adapter, get_include_paths
):
    unit_test = mock_unit_test_node()
    unit_test.overrides = UnitTestOverrides(macros={"macro_a": "override"})
    ctx = providers.generate_runtime_unit_test_context(
        unit_test=unit_test,
        config=config_postgres,
        manifest=manifest_fx,
    )
    assert ctx["macro_a"]() == "override"


def test_unit_test_runtime_context_macro_overrides_package(
    config_postgres, manifest_fx, get_adapter, get_include_paths
):
    unit_test = mock_unit_test_node()
    unit_test.overrides = UnitTestOverrides(macros={"some_package.some_macro": "override"})

    dbt_macro = mock_macro("some_macro", "some_package")
    manifest_with_dbt_macro = mock_manifest(config_postgres, additional_macros=[dbt_macro])

    ctx = providers.generate_runtime_unit_test_context(
        unit_test=unit_test,
        config=config_postgres,
        manifest=manifest_with_dbt_macro,
    )
    assert ctx["some_package"]["some_macro"]() == "override"


@pytest.mark.parametrize(
    "overrides,expected_override_value",
    [
        # override dbt macro at global level
        ({"some_macro": "override"}, "override"),
        # # override dbt macro at dbt-namespaced level level
        ({"dbt.some_macro": "override"}, "override"),
        # override dbt macro at both levels - global override should win
        (
            {"some_macro": "dbt_global_override", "dbt.some_macro": "dbt_namespaced_override"},
            "dbt_global_override",
        ),
        # override dbt macro at both levels - global override should win, regardless of order
        (
            {"dbt.some_macro": "dbt_namespaced_override", "some_macro": "dbt_global_override"},
            "dbt_global_override",
        ),
    ],
)
def test_unit_test_runtime_context_macro_overrides_dbt_macro(
    overrides,
    expected_override_value,
    config_postgres,
    manifest_fx,
    get_adapter,
    get_include_paths,
):
    unit_test = mock_unit_test_node()
    unit_test.overrides = UnitTestOverrides(macros=overrides)

    dbt_macro = mock_macro("some_macro", "dbt")
    manifest_with_dbt_macro = mock_manifest(config_postgres, additional_macros=[dbt_macro])

    ctx = providers.generate_runtime_unit_test_context(
        unit_test=unit_test,
        config=config_postgres,
        manifest=manifest_with_dbt_macro,
    )
    assert ctx["some_macro"]() == expected_override_value
    assert ctx["dbt"]["some_macro"]() == expected_override_value

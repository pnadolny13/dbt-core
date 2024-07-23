import pytest

from dbt.cli.main import dbtRunner
from dbt.contracts.graph.manifest import Manifest
from dbt.exceptions import TargetNotFoundError
from dbt.tests.util import get_manifest
from dbt_semantic_interfaces.type_enums import TimeGranularity
from tests.functional.metrics.fixtures import (
    bad_time_spine_yml,
    basic_metrics_yml,
    metricflow_time_spine_second_sql,
    metricflow_time_spine_sql,
    models_people_sql,
    semantic_model_people_yml,
    time_spines_yml,
)


class TestSuccessfulTimeSpines:
    @pytest.fixture(scope="class")
    def models(self):
        return {
            "basic_metrics.yml": basic_metrics_yml,
            "mf_time_spine_day.sql": metricflow_time_spine_sql,
            "mf_time_spine_second.sql": metricflow_time_spine_second_sql,
            "time_spines.yml": time_spines_yml,
            "semantic_model_people.yml": semantic_model_people_yml,
            "people.sql": models_people_sql,
        }

    def test_time_spines(self, project):
        runner = dbtRunner()
        result = runner.invoke(["parse"])
        assert result.success
        assert isinstance(result.result, Manifest)

        manifest = get_manifest(project.project_root)

        assert set(manifest.time_spines.keys()) == {
            "time_spine.test.time_spine_second",
            "time_spine.test.time_spine_day",
        }

        for time_spine in manifest.time_spines.values():
            assert time_spine.package_name == "test"
            assert time_spine.path == "time_spines.yml"
            assert time_spine.original_file_path == "models/time_spines.yml"

        time_spine_day = manifest.time_spines.get("time_spine.test.time_spine_day")
        time_spine_second = manifest.time_spines.get("time_spine.test.time_spine_second")
        assert time_spine_day.name == "time_spine_day"
        assert time_spine_second.name == "time_spine_second"
        assert time_spine_day.node_relation.alias == "mf_time_spine_day"
        assert time_spine_second.node_relation.alias == "mf_time_spine_second"
        assert time_spine_day.primary_column.name == "date_day"
        assert time_spine_second.primary_column.name == "ts_second"
        assert time_spine_day.primary_column.time_granularity == TimeGranularity.DAY
        assert time_spine_second.primary_column.time_granularity == TimeGranularity.SECOND


class TestTimeSpineModelDoesNotExist:
    @pytest.fixture(scope="class")
    def models(self):
        return {
            "basic_metrics.yml": basic_metrics_yml,
            "mf_time_spine_day.sql": metricflow_time_spine_sql,
            "mf_time_spine_second.sql": metricflow_time_spine_second_sql,
            "time_spines.yml": bad_time_spine_yml,
            "semantic_model_people.yml": semantic_model_people_yml,
            "people.sql": models_people_sql,
        }

    def test_time_spines(self, project):
        runner = dbtRunner()
        result = runner.invoke(["parse"])
        assert not result.success

        # Bad model ref in time spine def
        assert isinstance(result.exception, TargetNotFoundError)
        assert (
            "Time_Spine 'time_spine.test.bad_model_ref' (models/time_spines.yml) depends on a node named 'doesnt_exist' which was not found"
            in result.exception.msg
        )


# TODO: test legacy time spine


class TestLegacyTimeSpine:
    @pytest.fixture(scope="class")
    def models(self):
        return {
            "basic_metrics.yml": basic_metrics_yml,
            "metricflow_time_spine.sql": metricflow_time_spine_sql,
            "semantic_model_people.yml": semantic_model_people_yml,
            "people.sql": models_people_sql,
        }

    def test_time_spines(self, project):
        runner = dbtRunner()
        result = runner.invoke(["parse"])
        assert result.success
        assert isinstance(result.result, Manifest)

        manifest = get_manifest(project.project_root)

        # assert manifest.time_spines.keys()) == {
        #     "time_spine.test.time_spine_sset(econd",
        #     "time_spine.test.time_spine_day",
        # }

        # for time_spine in manifest.time_spines.values():
        #     assert time_spine.package_name == "test"
        #     assert time_spine.path == "time_spines.yml"
        #     assert time_spine.original_file_path == "models/time_spines.yml"


# also failure case where neither exists

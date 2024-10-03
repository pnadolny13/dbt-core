import pytest
import yaml
from dbt_config.catalog_config import ExternalCatalog

from dbt.tests.util import run_dbt, write_file
from tests.fixtures.jaffle_shop import JaffleShopProject


@pytest.fixture(scope="class", autouse=True)
def dbt_catalog_config(project_root):
    config = {
        "catalogs": [
            {
                "name": "my_external_catalog",
                "type": "iceberg",
                "configuration": {
                    "table_format": "parquet",
                    "namespace": "dbt",
                    "external_location": "s3://my-bucket/my-path",
                },
                "management": {
                    "enabled": True,
                    "create_if_not_exists": False,
                    "alter_if_different": False,
                    "read_only": True,
                    "refresh": "on_change",
                },
            }
        ],
    }
    write_file(yaml.safe_dump(config), project_root, "catalog.yml")


class TestCatalogConfig(JaffleShopProject):

    def test_supplying_external_catalog(self, project):
        manifest = run_dbt(["parse"])
        assert manifest.catalogs != {}
        ExternalCatalog.model_validate_json(manifest.catalogs["my_external_catalog"])

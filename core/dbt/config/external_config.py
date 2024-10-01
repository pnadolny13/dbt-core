from typing import Optional

from dbt_config.external_config import ExternalCatalogConfig

from dbt.clients.yaml_helper import load_yaml_text
from dbt.constants import EXTERNAL_CATALOG_FILE_NAME
from dbt_common.clients.system import load_file_contents, path_exists


def _load_yaml(path):
    contents = load_file_contents(path)
    return load_yaml_text(contents)


def _load_yml_dict(file_path):
    if path_exists(file_path):
        ret = _load_yaml(file_path) or {}
        return ret
    return None


def load_external_catalog_config(project_root) -> Optional[ExternalCatalogConfig]:
    unparsed_config = _load_yml_dict(f"{project_root}/{EXTERNAL_CATALOG_FILE_NAME}")
    if unparsed_config is not None:
        return ExternalCatalogConfig.model_validate(unparsed_config)
    return None

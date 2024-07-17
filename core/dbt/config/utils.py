from typing import Any, Dict

from dbt.clients import yaml_helper
from dbt.events.types import InvalidOptionYAML
from dbt.exceptions import DbtExclusivePropertyUseError, OptionNotYamlDictError
from dbt_common.events.functions import fire_event
from dbt_common.exceptions import DbtValidationError


def parse_cli_vars(var_string: str) -> Dict[str, Any]:
    return parse_cli_yaml_string(var_string, "vars")


def parse_cli_yaml_string(var_string: str, cli_option_name: str) -> Dict[str, Any]:
    try:
        cli_vars = yaml_helper.load_yaml_text(var_string)
        var_type = type(cli_vars)
        if var_type is dict:
            return cli_vars
        else:
            raise OptionNotYamlDictError(var_type, cli_option_name)
    except (DbtValidationError, OptionNotYamlDictError):
        fire_event(InvalidOptionYAML(option_name=cli_option_name))
        raise


def normalize_warn_error_options(
    dictionary: Dict[str, Any],
) -> None:
    """Fixes fields for warn_error_options from yaml format to fields
    expected by the WarnErrorOptions class.
    'error' => 'include', 'warn' => 'exclude'

    Also validates that two different forms of accepted keys are not
    both provided.
    """

    if "include" in dictionary and "error" in dictionary:
        raise DbtExclusivePropertyUseError(
            "Only `error` or `include` can be specified in `warn_error_options`, not both"
        )

    if "warn" in dictionary and "exclude" in dictionary:
        raise DbtExclusivePropertyUseError(
            "Only `warn` or `exclude` can be specified in `warn_error_options`, not both"
        )

    convert = {
        "error": "include",
        "warn": "exclude",
    }
    for key in list(convert.keys()):
        if key in dictionary:
            value = dictionary.pop(key)
            if value is None:
                value = []
            dictionary[convert[key]] = value
    if "silence" in dictionary and dictionary["silence"] is None:
        dictionary["silence"] = []

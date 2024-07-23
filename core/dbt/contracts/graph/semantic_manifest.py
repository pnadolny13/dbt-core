from typing import List, Optional

from dbt.constants import LEGACY_TIME_SPINE_MODEL_NAME
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import ManifestNode, TimeSpine
from dbt.events.types import SemanticValidationFailure
from dbt.exceptions import ParsingError
from dbt_common.clients.system import write_file
from dbt_common.events.base_types import EventLevel
from dbt_common.events.functions import fire_event
from dbt_semantic_interfaces.implementations.metric import PydanticMetric
from dbt_semantic_interfaces.implementations.node_relation import PydanticNodeRelation
from dbt_semantic_interfaces.implementations.project_configuration import (
    PydanticProjectConfiguration,
)
from dbt_semantic_interfaces.implementations.saved_query import PydanticSavedQuery
from dbt_semantic_interfaces.implementations.semantic_manifest import (
    PydanticSemanticManifest,
)
from dbt_semantic_interfaces.implementations.semantic_model import PydanticSemanticModel
from dbt_semantic_interfaces.implementations.time_spine import (
    PydanticTimeSpine,
    PydanticTimeSpinePrimaryColumn,
)
from dbt_semantic_interfaces.implementations.time_spine_table_configuration import (
    PydanticTimeSpineTableConfiguration,
)
from dbt_semantic_interfaces.type_enums import TimeGranularity
from dbt_semantic_interfaces.validations.semantic_manifest_validator import (
    SemanticManifestValidator,
)


class SemanticManifest:
    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest

    def validate(self) -> bool:

        # TODO: Enforce this check.
        # if self.manifest.metrics and not self.manifest.semantic_models:
        #    fire_event(
        #        SemanticValidationFailure(
        #            msg="Metrics require semantic models, but none were found."
        #        ),
        #        EventLevel.ERROR,
        #    )
        #    return False

        if not self.manifest.metrics or not self.manifest.semantic_models:
            return True

        semantic_manifest = self._get_pydantic_semantic_manifest()
        validator = SemanticManifestValidator[PydanticSemanticManifest]()
        validation_results = validator.validate_semantic_manifest(semantic_manifest)

        for warning in validation_results.warnings:
            fire_event(SemanticValidationFailure(msg=warning.message))

        for error in validation_results.errors:
            fire_event(SemanticValidationFailure(msg=error.message), EventLevel.ERROR)

        return not validation_results.errors

    def write_json_to_file(self, file_path: str):
        semantic_manifest = self._get_pydantic_semantic_manifest()
        json = semantic_manifest.json()
        write_file(file_path, json)

    def _get_pydantic_semantic_manifest(self) -> PydanticSemanticManifest:
        time_spines = list(self.manifest.time_spines.values())

        pydantic_time_spines: List[PydanticTimeSpine] = []
        daily_time_spine: Optional[TimeSpine] = None
        for time_spine in time_spines:
            # Assertion for type checker
            assert time_spine.node_relation, (
                f"Node relation should have been set for time time spine {time_spine.name} during "
                "manifest parsing, but it was not."
            )
            pydantic_time_spine = PydanticTimeSpine(
                name=time_spine.name,
                node_relation=PydanticNodeRelation(
                    alias=time_spine.node_relation.alias,
                    schema_name=time_spine.node_relation.schema_name,
                    database=time_spine.node_relation.database,
                    relation_name=time_spine.node_relation.relation_name,
                ),
                primary_column=PydanticTimeSpinePrimaryColumn(
                    name=time_spine.primary_column.name,
                    time_granularity=time_spine.primary_column.time_granularity,
                ),
            )
            pydantic_time_spines.append(pydantic_time_spine)
            if time_spine.primary_column.time_granularity == TimeGranularity.DAY:
                daily_time_spine = time_spine

        project_config = PydanticProjectConfiguration(
            time_spine_table_configurations=[], time_spines=pydantic_time_spines
        )
        pydantic_semantic_manifest = PydanticSemanticManifest(
            metrics=[], semantic_models=[], project_configuration=project_config
        )

        for semantic_model in self.manifest.semantic_models.values():
            pydantic_semantic_manifest.semantic_models.append(
                PydanticSemanticModel.parse_obj(semantic_model.to_dict())
            )

        for metric in self.manifest.metrics.values():
            pydantic_semantic_manifest.metrics.append(PydanticMetric.parse_obj(metric.to_dict()))

        for saved_query in self.manifest.saved_queries.values():
            pydantic_semantic_manifest.saved_queries.append(
                PydanticSavedQuery.parse_obj(saved_query.to_dict())
            )

        if self.manifest.semantic_models:
            # Validate that there is a time spine configured for the semantic manifest.

            # If no daily time spine has beem configured, look for legacy time spine model. This logic is included to
            # avoid breaking projects that have not migrated to the new time spine config yet.
            legacy_time_spine_model: Optional[ManifestNode] = None
            if not daily_time_spine:
                legacy_time_spine_model = self.manifest.ref_lookup.find(
                    LEGACY_TIME_SPINE_MODEL_NAME, None, None, self.manifest
                )
                # If no time spines have been configured AND legacy time spine model does not exist, error.
                if not legacy_time_spine_model:
                    raise ParsingError(
                        "The semantic layer requires a time spine model in the project, but none was found. "
                        "Guidance on creating this model can be found on our docs site ("
                        "https://docs.getdbt.com/docs/build/metricflow-time-spine) "  # TODO: update docs link!
                    )
                # Create time_spine_table_config, set it in project_config, and add to semantic manifest
                time_spine_table_config = PydanticTimeSpineTableConfiguration(
                    location=legacy_time_spine_model.relation_name,
                    column_name="date_day",
                    grain=TimeGranularity.DAY,
                )
                pydantic_semantic_manifest.project_configuration.time_spine_table_configurations = [
                    time_spine_table_config
                ]

        return pydantic_semantic_manifest

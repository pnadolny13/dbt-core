import time
from dataclasses import dataclass, field
from typing import List, Optional

from dbt.artifacts.resources.base import GraphResource
from dbt.artifacts.resources.v1.components import DependsOn, RefArgs
from dbt_common.dataclass_schema import dbtClassMixin
from dbt.artifacts.resources.v1.semantic_model import NodeRelation
from dbt_semantic_interfaces.type_enums.time_granularity import TimeGranularity

# ====================================
# TimeSpine objects
# TimeSpine protocols: https://github.com/dbt-labs/dbt-semantic-interfaces/blob/main/dbt_semantic_interfaces/protocols/time_spine.py
# ====================================


@dataclass
class TimeSpinePrimaryColumn(dbtClassMixin):
    """The column in the time spine that maps to a standard granularityf."""

    name: str
    time_granularity: TimeGranularity


@dataclass
class TimeSpine(GraphResource):
    """Describes a table that contains dates at a specific time grain.

    One column must map to a standard granularity (one of the TimeGranularity enum members). Others might represent
    custom granularity columns. Custom granularity columns are not yet implemented.
    """

    model: str
    node_relation: Optional[NodeRelation]
    primary_column: TimeSpinePrimaryColumn
    depends_on: DependsOn = field(default_factory=DependsOn)
    refs: List[RefArgs] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())

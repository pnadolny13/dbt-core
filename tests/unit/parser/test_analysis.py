import pytest

from dbt.artifacts.resources.types import NodeType
from dbt.config.runtime import RuntimeConfig
from dbt.contracts.files import FileHash, FilePath, SourceFile
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.model_config import NodeConfig
from dbt.contracts.graph.nodes import AnalysisNode, DependsOn
from dbt.parser.analysis import AnalysisParser
from dbt.parser.search import FileBlock
from tests.unit.utils import normalize


class TestAnalysisParser:

    @pytest.fixture
    def analysis_parser(self, manifest: Manifest, runtime_config: RuntimeConfig) -> AnalysisParser:
        return AnalysisParser(
            project=runtime_config,
            manifest=manifest,
            root_project=runtime_config,
        )

    @pytest.fixture
    def file_block(self, runtime_config: RuntimeConfig) -> FileBlock:
        raw_code = "SELECT 'misu' AS best_cat"
        file_path = FilePath(
            searched_path="",
            relative_path="",
            modification_time=0,
            project_root="",
        )

        source_file = SourceFile(
            path=file_path,
            checksum=FileHash.from_contents(raw_code),
            project_name=runtime_config.project_name,
            contents=raw_code,
        )
        return FileBlock(file=source_file)

    def test_basic(
        self, manifest: Manifest, analysis_parser: AnalysisParser, file_block: FileBlock
    ):
        manifest.files[file_block.file.file_id] = file_block.file
        analysis_parser.parse_file(file_block)
        breakpoint()
        expected = AnalysisNode(
            alias="analysis_1",
            name="analysis_1",
            database="test",
            schema="analytics",
            resource_type=NodeType.Analysis,
            unique_id="analysis.snowplow.analysis_1",
            fqn=["snowplow", "analysis", "nested", "analysis_1"],
            package_name="snowplow",
            original_file_path=normalize("analyses/nested/analysis_1.sql"),
            depends_on=DependsOn(),
            config=NodeConfig(),
            path=normalize("analysis/nested/analysis_1.sql"),
            language="sql",
            raw_code=file_block.file.contents or "",
            checksum=file_block.file.checksum,
            unrendered_config={},
            relation_name=None,
        )
        node = list(analysis_parser.manifest.nodes.values())[0]
        assert node == expected
        file_id = "snowplow://" + normalize("analyses/nested/analysis_1.sql")
        assert file_id in analysis_parser.manifest.files
        file = analysis_parser.manifest.files[file_id]
        assert isinstance(file, SourceFile)
        assert file.nodes == ["analysis.snowplow.analysis_1"]

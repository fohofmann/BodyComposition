"""Optional final reporting stage case-report action."""

from __future__ import annotations

from pathlib import Path

from BodyComposition.pipeline import PipelineAction
from BodyComposition.reporting.contracts import CaseReportInput, ReportingSettings
from BodyComposition.reporting.service import render_case_report


CASE_REPORT_PDF = "reports/{caseid}/case_report.pdf"
CASE_REPORT_MANIFEST = "reports/{caseid}/report_manifest.json"


class RenderCaseReport(PipelineAction):
    """Render one derived case page after authoritative measurement stage export."""

    licenses = ["reportlab", "pypdf", "bitstream_vera"]

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.settings = ReportingSettings.from_mapping(self.config)
        if not self.settings.enabled:
            raise ValueError("RenderCaseReport cannot be constructed while reporting is disabled.")
        self.io_inputs = [
            "tmp/prepared_image",
            "tmp/orientation_result",
            "tmp/vertebral_result",
            "tmp/measurement_bundle",
        ]
        self.io_outputs = [CASE_REPORT_PDF, CASE_REPORT_MANIFEST, "tmp/report_result"]
        self.io_persisted_outputs = [CASE_REPORT_PDF, CASE_REPORT_MANIFEST]
        self.io_reset_outputs = [CASE_REPORT_PDF, CASE_REPORT_MANIFEST]

    def __call__(self, memory):
        super().__call__(memory)
        case_id = str(memory["id"])
        report_input = CaseReportInput(
            case_id=case_id,
            prepared_image=memory["tmp/prepared_image"].prepared_image,
            vertebral_result=memory["tmp/vertebral_result"],
            measurement_bundle=memory["tmp/measurement_bundle"],
            orientation=memory["tmp/orientation_result"],
        )
        output_directory = Path(memory["workspace"]) / "reports" / case_id
        result = render_case_report(report_input, output_directory, self.settings)
        memory[CASE_REPORT_PDF] = result.pdf_path
        memory[CASE_REPORT_MANIFEST] = result.manifest_path
        memory["tmp/report_result"] = result

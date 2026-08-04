# Documentation

Start with the repository [README](../README.md) for installation and a first
run. Use the pages below when a task needs more detail.

| Task | Guide |
| --- | --- |
| Understand the stages and physical-coordinate contract | [Pipeline and geometry](pipeline.md) |
| Synchronize, verify, select, and cite models | [Models and licensing](models.md) |
| Analyze DICOM or convert it for transfer | [DICOM input and conversion](dicom.md) |
| Run the L3-only preset | [Low-resource analysis](low-resource.md) |
| Generate or validate configuration | [Configuration](config.md) |
| Run batches, multiple GPUs, or Slurm arrays | [Parallel execution](execution.md) |
| Build or run a container | [Containers](containers.md) |
| Understand result folders and manifests | [Output schema](output-schema.md) |
| Interpret measurement tables and signatures | [Measurements](measurements.md) |
| Distinguish compartments, tissues, and labels | [Labels](labels.md) and [tissue definitions](tissue_definitions.md) |
| Review warnings and failures | [QC and manual review](qc-review.md) |
| Enable or export PDF reports | [Reporting](reporting.md) |
| Freeze and archive a cohort analysis | [Reproducibility](reproducibility.md) |
| Compare observed runtime and memory | [Performance](performance.md) |
| Diagnose setup and execution problems | [Troubleshooting](troubleshooting.md) |
| Review intended use and limitations | [Known limitations](known-limitations.md) |

The code-generated defaults and packaged JSON schemas remain authoritative
when prose and an installed release disagree:

```bash
bodycomposition config show-default -o bodycomposition.yaml
bodycomposition config validate bodycomposition.yaml --json
bodycomposition version --json
```

For contributions and release checks, see [CONTRIBUTING.md](../CONTRIBUTING.md).
For third-party attribution and model-specific citation requirements, see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

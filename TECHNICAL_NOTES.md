# Technical Notes

## Runtime Components

- `apps/rna_workbench/server.py` runs the local HTTP server and API.
- `apps/rna_workbench/static/index.html` is the browser UI.
- `trees/run_matrixmodel_seq2seq_ar_designer_torch.py` defines the model class and condition encoders needed to load the checkpoints.
- `trees/rna_models.py` contains model position embeddings and shared model components.
- `trees/rnaplot_design_highlight.py` styles and annotates ViennaRNA SVG output.

## API Endpoints

```text
GET  /api/status
POST /api/design
POST /api/random_example
POST /api/mask_constraints
POST /api/sample_structures
POST /api/render
```

## Checkpoints

The app expects these files by default:

```text
checkpoints/full_structure_small.pt
checkpoints/fsb_partial_base_small.pt
```

SHA256:

```text
a28e650ba0a8fd61a92ade424d939f3d95631df63f6f9f2ae78a5f81932b472f  checkpoints/full_structure_small.pt
38e7a884c26976e003c50fac290d7bf8d636fa13c645a348f47f9fe86a3de09d  checkpoints/fsb_partial_base_small.pt
```

Override checkpoint paths with:

```bash
RNA_WORKBENCH_FS_SMALL=/path/to/full_structure_small.pt
RNA_WORKBENCH_FSB_SMALL=/path/to/fsb_partial_base_small.pt
```

## Constraint Behavior

- Full-structure-only targets use `pretrained_small`.
- Partial structure targets or concrete base constraints use `fsb_pretrained_small`.
- Structure input accepts dot-bracket `.()`, unknown structure `?`, and paired-unknown `#`.
- Base mask accepts concrete bases plus ambiguity/unconstrained tokens such as `?`, `N`, and `#`.
- `struct err` and `cond err` are MFE-based diagnostics.
- `p` and `log p/nt` are ensemble/Boltzmann diagnostics.

## Implementation Notes

- Random structure masking is pair-aware.
- Partial target rendering uses a balanced dot-bracket skeleton for SVG layout.
- ViennaRNA SVG rendering runs in a child process so an RNAplot failure does not kill the main server.
- UI requests have timeouts so the browser reports backend failures instead of waiting indefinitely.


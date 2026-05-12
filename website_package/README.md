# RNA Workbench Website Package

This package contains the static website folder:

```text
wasm/
  index.html
  config.js
```

Copy `wasm/` into your website and serve it as a normal static directory, for example:

```text
https://your-site.example/wasm/
```

## Configure The Backend

The current RNA Workbench model runtime is still the Python/PyTorch backend. Edit `wasm/config.js` to point the static UI at that backend:

```js
window.RNA_WORKBENCH_API_BASE = "https://rna-api.your-site.example";
```

Leave it empty only when `/api/...` is served from the same origin as the page:

```js
window.RNA_WORKBENCH_API_BASE = "";
```

For a reverse proxy mounted at `/rna-workbench-api`, use:

```js
window.RNA_WORKBENCH_API_BASE = "/rna-workbench-api";
```

The UI calls:

```text
/api/status
/api/design
/api/random_example
/api/mask_constraints
/api/sample_structures
/api/render
```

## Run The Backend

From the original `rna_workbench_minimal` directory:

```bash
RNA_WORKBENCH_PYTHON=.venv/bin/python \
RNA_WORKBENCH_DEVICE=cpu \
RNA_WORKBENCH_HOST=0.0.0.0 \
RNA_WORKBENCH_PORT=7861 \
RNA_WORKBENCH_CORS_ORIGIN=https://your-site.example \
bash run_workbench.sh
```

For local testing from any origin, you can use:

```bash
RNA_WORKBENCH_CORS_ORIGIN='*'
```

## About WASM

This is a website-ready package, not a complete in-browser WASM port. The current backend depends on PyTorch inference, ViennaRNA, and two roughly 680 MB checkpoint files. A true browser-only WASM/WebGPU build would require a separate model export/runtime project and likely a different packaging strategy for the checkpoint weights.

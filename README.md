# CrossPascal Converter

`crosspascal-convert` contains 26 Python files selected from the planned standalone model converter. This private testing draft lets you review the architecture and source. The files remain under `source/pascalvision_convert/`, and the Python package name remains `pascalvision_convert`.

**It is not an installable Python package.** Required family and support modules are omitted. A normal package import reaches missing dependencies, including `tuning`. This snapshot does not provide a supported conversion command or model download.

## Verify the source

From the repository root, use Python 3.12 or later:

```powershell
python -B scripts/verify_source.py
```

The checker uses only the standard library. It verifies this snapshot without importing the converter or loading ML dependencies. Root Git metadata is declared separately; unexpected payload or build files cause a failure.

## Intended product

The full converter is planned to provide a CLI and wizard, explicit build plans, model recipes, target selection, and numerical validation. ONNX/TFLite support and each model/backend route require their own complete inputs and acceptance.

See [source notes](docs/SOURCE-NOTES.md) for the constraints preserved during source preparation. The source and its recorded status data may describe historical experiments; they do not expand this snapshot's supported functions.

## License

The original source, checker, and documentation use the [MIT license](LICENSE). External Python packages, native runtimes, pretrained weights, and test models are not included.

# Audio CNN — Environmental Sound Classification (ESC-50)

A ResNet-style convolutional neural network that classifies environmental sounds
from the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) into 50 categories.
The model operates on spectrogram representations of audio clips and is trained
on a cloud GPU using [Modal](https://modal.com).

---

## Overview

|                       |                                                                                |
| --------------------- | ------------------------------------------------------------------------------ |
| **Task**              | Multi-class audio classification (50 classes)                                  |
| **Dataset**           | ESC-50 — 2,000 labeled 5-second clips across 50 environmental sound categories |
| **Input**             | Single-channel spectrograms (treated as 1×H×W images)                          |
| **Model**             | ResNet-34-style CNN (`AudioCNN`) built from residual blocks                    |
| **Augmentation**      | Mixup                                                                          |
| **Training platform** | Modal (`A10G` GPU)                                                             |
| **Logging**           | TensorBoard event files, persisted to a Modal Volume                           |
| **Artifact**          | `best_model.pth` (~81 MiB), persisted to a Modal Volume                        |

The pipeline downloads and bakes ESC-50 into the container image, mounts persistent
volumes for the dataset and model outputs, trains the network, logs metrics to
TensorBoard, and saves the best checkpoint.

---

## Architecture

`AudioCNN` follows the classic ResNet-34 layout — a stem followed by four stages of
residual blocks with `[3, 4, 6, 3]` blocks respectively, then global pooling and a
linear classifier.

```
Input  (1 × H × W spectrogram)
  │
  ├─ Stem:  Conv2d(1→64, 7×7, stride 2) → BatchNorm → ReLU → MaxPool(3×3, stride 2)
  │
  ├─ layer1:  3 × ResidualBlock(64  → 64)
  ├─ layer2:  4 × ResidualBlock(64  → 128, first block stride 2)
  ├─ layer3:  6 × ResidualBlock(128 → 256, first block stride 2)
  ├─ layer4:  3 × ResidualBlock(256 → 512, first block stride 2)
  │
  ├─ AdaptiveAvgPool2d(1×1)
  ├─ Dropout(0.5)
  └─ Linear(512 → num_classes)
```

### Residual block

Each `ResidualBlock` is two 3×3 convolutions with batch norm, plus a skip connection.
When the block changes channel count or spatial resolution (stride ≠ 1), the skip
path uses a 1×1 convolution so the shapes match before the element-wise add.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.use_shortcut = stride != 1 or in_channels != out_channels
        self.shortcut = nn.Sequential()
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x) if self.use_shortcut else x
        return torch.relu(out + shortcut)
```

> **Note:** `conv1`'s stride must equal the block's `stride` (not a hardcoded value),
> otherwise the main path and the shortcut downsample by different factors and the
> `out + shortcut` add fails with a size mismatch.

---

## Project structure

```
.
├── train.py            # Modal app: image definition, dataset, training loop, entrypoint
├── model.py            # AudioCNN and ResidualBlock definitions
├── requirements.txt    # Python dependencies installed into the Modal image
└── README.md
```

---

## Modal resources

Two persistent [Modal Volumes](https://modal.com/docs/guide/volumes) are used:

| Volume       | Mount point | Purpose                                                     |
| ------------ | ----------- | ----------------------------------------------------------- |
| `esc50-data` | `/data`     | Dataset storage                                             |
| `esc-model`  | `/models`   | Trained checkpoints (`best_model.pth`) and TensorBoard logs |

The ESC-50 dataset is downloaded during image build and placed at `/opt/esc50-data`
inside the container.

### Image definition

```python
import modal

app = modal.App("audio-cnn-model")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
    .run_commands([
        "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
        "cd /tmp && unzip esc50.zip",
        "mkdir -p /opt/esc50-data",
        "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
        "rm -rf /tmp/esc50.zip /tmp/ESC-50-master",
    ])
    .add_local_python_source("model")
)

volume       = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume, "/models": model_volume},
    timeout=60 * 60 * 3,   # 3 hours
)
def train():
    ...


@app.local_entrypoint()
def main():
    train.remote()
```

---

## Requirements

- Python 3.12
- A [Modal](https://modal.com) account (`pip install modal`, then `modal setup`)
- Dependencies in `requirements.txt` (installed automatically inside the Modal image)

For **local** work (viewing logs, running the entrypoint), install into a virtual
environment or conda env:

```bash
conda create -n audiocnn python=3.12
conda activate audiocnn
pip install modal tensorboard
```

> **NumPy note:** if you `import torch` locally you may see a warning about torch being
> compiled against NumPy 1.x while NumPy 2.x is installed. It's harmless for the Modal
> run (the container builds its own environment), but to silence it locally either pin
> `numpy<2` or move heavy imports (`torch`, etc.) inside the remote function / an
> `image.imports()` block so they never import on your Mac.

---

## Running training

Authenticate once:

```bash
modal setup
```

Then launch training. Use `--detach` so the run survives a dropped connection or an
accidental `Ctrl+C` (the first run is slow because it builds the image and downloads
the dataset):

```bash
modal run --detach train.py
```

Progress and a live dashboard link are printed to the terminal, and metrics are written
to TensorBoard event files in the `esc-model` volume.

---

## Retrieving results

List what's in the model volume:

```bash
modal volume ls esc-model
```

Download the trained checkpoint:

```bash
modal volume get esc-model /best_model.pth ./best_model.pth
```

Download the TensorBoard logs. **Important:** pass a destination _directory_ that already
exists (with a trailing slash), otherwise `modal volume get` writes the file _as_ the
destination path rather than _into_ it:

```bash
mkdir -p ./logs
modal volume get esc-model /tensorboard_logs ./logs/
tensorboard --logdir ./logs
```

Then open <http://localhost:6006>.

---

## Viewing TensorBoard

```bash
tensorboard --logdir ./logs
```

Point `--logdir` at the **parent** folder that contains all your run subfolders — TensorBoard
scans recursively and overlays every run on the same charts, which makes comparing runs easy.
If a previous TensorBoard instance is still running, stop it (`Ctrl+C`) or start the new one
on a different port with `--port 6007`.

---

## Troubleshooting

Gotchas encountered while building this project, and their fixes:

**`modal run` fails with "can't find Rust compiler" (installing `modal`)**
A dependency (`cbor2`) tried to build from source. Upgrade pip so it uses a prebuilt wheel:
`pip install --upgrade pip`. Or install a pure-Python cbor2: `CBOR2_BUILD_C_EXTENSION=0 pip install "cbor2<6"`.

**`cp: target '/opt/...': No such file or directory` during image build**
The `mkdir` and `cp` target paths must match exactly — same name, both absolute (leading `/`).

**`TypeError: 'method' object is not iterable` / `unsupported operand for +=`**
A pandas/torch method was referenced without calling it. Add the parentheses:
`.unique()` not `.unique`, `.item()` not `.item`.

**`Conv2d.__init__() got multiple values for argument 'stride'`**
`stride` was passed both positionally and as a keyword. Don't pass `self` to `nn.Conv2d`,
and don't duplicate the `stride` argument.

**`The size of tensor a (…) must match tensor b (…)` in a residual add**
`conv1`'s stride didn't match the shortcut's stride. Use `stride=stride` in the block's
first conv.

**`expected input[…] to have N channels, but got M`**
The `forward` method ran a stage in the wrong order or twice. Run each stage once:
`layer1 → layer2 → layer3 → layer4`.

**TensorBoard shows "No dashboards are active" after downloading logs**
Usually the download landed the event file _as_ a plain file instead of _inside_ a
directory. TensorBoard needs a directory to scan — move the file into one:
`mkdir -p clean_logs/run1 && mv <downloaded_file> clean_logs/run1/ && tensorboard --logdir clean_logs`.

**Log folders have weird names with an invisible carriage return (e.g. `tesorboard_logs⏎run_...`)**
This comes from a backslash in the log-dir string: `"tensorboard_logs\run_..."` — Python
reads `\r` as a carriage return. Use a forward slash: `f"tensorboard_logs/run_{timestamp}"`.
This also nests runs neatly under one parent folder.

---

## Notes & possible improvements

- **Sanity-check the loss.** For 50 classes, chance-level cross-entropy is `ln(50) ≈ 3.91`.
  A loss near that after epoch 1 is normal; it should fall steadily as training proceeds.
- **Use the official folds.** ESC-50 ships with 5 predefined cross-validation folds. Reporting
  accuracy across those folds (rather than a single random 1600/400 split) makes results
  comparable to published baselines.
- **Consider transfer learning.** ESC-50 is small (2,000 clips), so a from-scratch ResNet-34
  overfits quickly. Starting from ImageNet-pretrained weights on the spectrogram "images"
  typically outperforms training from scratch on datasets this size. Mixup (already used here)
  also helps regularize.
- **Lint before you launch.** Most bugs above are typos a linter/type-checker (ruff, pyright)
  catches instantly. A quick local smoke test with a dummy tensor before each Modal run saves
  a full image rebuild per mistake.

---

## License

The ESC-50 dataset is distributed under its own license — see the
[ESC-50 repository](https://github.com/karolpiczak/ESC-50) for details.
Add a license for your own code as appropriate.

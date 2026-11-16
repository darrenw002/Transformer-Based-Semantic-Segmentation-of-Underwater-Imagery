# UNetFormer source and SUIM adaptation

- Official repository: https://github.com/WangLibo1995/GeoSeg
- Pinned commit: `9453fe48209c4626b29e35e61bab93b61212c4b1`.
- Original model: `geoseg/models/UNetFormer.py`.
- Original model-file SHA-256: `b20244bc02c345f7a9948d920fab02ba6f3605ea055400805cb58c0feeecaa7a`.
- Architecture classes are included directly in `notebooks/04_unetformer_baseline.ipynb`.
- GeoSeg's GPL-3.0 license is preserved in `LICENSE-GeoSeg` and reproduced in the notebook.
- Source credit: Libo Wang and coauthors, UNetFormer/GeoSeg.

Two compatibility edits preserve the intended operations with current PyTorch:
horizontal reflection padding uses a four-value tuple for a four-dimensional
tensor, and `meshgrid` explicitly specifies `indexing='ij'`. Imports use
`timm.layers`. Model widths and attention/decoder structure are retained.

The notebook uses torchvision's ImageNet-1k ResNet-18 V1 weights, downloaded from
`https://download.pytorch.org/models/resnet18-f37072fd.pth` with hash checking.
This replaces the original default SWSL-pretrained ResNet-18, avoiding its extra
pretraining data in the comparison with our ImageNet-pretrained SegFormer.
Checkpoint loading creates the backbone with `pretrained=False` and loads all
SUIM weights locally.

For the SUIM experiment:

- Input/target geometry remains 320 x 240. The forward wrapper reflects bottom
  and right edges as needed to reach a multiple of 32, then crops both prediction
  heads back before loss or scoring.
- Five independent foreground channels use the same RGB conversion, split,
  augmentation, and evaluation code as the SegFormer notebook.
- Training uses main-head BCE plus 0.4 times auxiliary BCE. Validation uses
  main-head BCE only. This replaces the original multiclass CE/Dice loss.
- Batch size 2, 50 epochs, AdamW at 6e-5, weight decay 0.01, and validation-loss
  checkpoint selection match our SegFormer budget. We do not reproduce GeoSeg's
  differential learning rates, Lookahead optimizer, or cosine schedule.
- Outputs reuse `data/test/output/unetformer_resnet18/`. Weights, training
  history, and loss plots reuse `ckpt/unetformer_resnet18/`.

These are a documented SUIM adaptation, rather than a reproduction of the
remote-sensing paper's original training results. Original SUIM dataset files
are unchanged. Notebook metadata records the original source, adapted
architecture-cell hash, shared metric-cell hash, and validation evidence.

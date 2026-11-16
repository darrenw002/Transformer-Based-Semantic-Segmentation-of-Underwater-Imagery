# Upstream SUIM source

- Repository: https://github.com/xahidbuffon/SUIM
- Imported commit: [59f7cb0d4d13289434ad6e3ab162ab7a5dcc5c84](https://github.com/xahidbuffon/SUIM/commit/59f7cb0d4d13289434ad6e3ab162ab7a5dcc5c84)
- Imported directories: `models/` and `utils/` at the project root.
- Imported entrypoints: all six `train_*.py` / `test_*.py` pairs and `get_f1_iou.py`, also at the project root.
- License: the upstream MIT license is preserved verbatim in `LICENSE-SUIM`.
- Every original download was verified against its upstream Git blob hash. Local output-path changes are listed below; model and utility source files are unchanged.
- Source import checks: Git blob hashes and Python syntax. Entrypoint scripts have not been executed directly.

## Why keep both complete directories?

For the full collection of benchmark models, keeping both directories preserves
the upstream package layout and the shared architecture helpers. A single model
can use a smaller subset after its imports are traced. `utils/data_utils.py`
provides the training data generator, and `utils/measure_utils.py` provides
evaluation functions.

The model and utility directories and all 13 training/test/evaluation entrypoints
are now present. They were imported from the same upstream revision; the local
output configuration changes below were then applied to the test/evaluation scripts.
Legacy Python/Keras compatibility and local data/checkpoint paths still need
to be addressed before running those experiments.

The separate [`notebooks/02_unet_baseline.ipynb`](notebooks/02_unet_baseline.ipynb)
includes compatible U-Net adaptations directly in its cells. Its architecture
and boundary-score function were copied from these upstream sources; label
conversion, loading, training, prediction, and evaluation were adapted from the
corresponding utilities and entrypoints. Source hashes are also recorded in the
notebook metadata. The notebook documents training-protocol differences and
defaults to the published U-Net checkpoint. Notebook metadata records current
source hashes in `suim_sources.sha256`, the original imported hashes in
`suim_sources.upstream_sha256`, and local changes for the referenced scripts.

## Local output configuration

Each test script writes to `data/test/output/<model>/`, reusing that model's
folder on subsequent runs:

| Script | Model folder |
| --- | --- |
| `test_unet.py` | `unet/` |
| `test_suimnet.py` | `suimnet/` |
| `test_segnet.py` | `segnet/` |
| `test_pspnet.py` | `pspnet/` |
| `test_fcn8.py` | `fcn8/` |
| `test_deeplab.py` | `deeplab/` |

The U-Net notebook also writes to `data/test/output/unet/`, including its score
reports and preview. DeepLab and FCN8 fish prediction folders use `FV/` instead
of upstream's `FB/`, matching the other models and the ground-truth category name.
`get_f1_iou.py` has a `model_name` setting, defaulting to `unet`, to select the
prediction folder. No model, training, or metric calculations were changed by
these output configuration edits.

## Repeat the import in another project (PowerShell + Git)

Run from your project root with the destination `models/`, `utils/`, license,
and entrypoint files absent, and no existing `SUIM-upstream/` checkout,
so existing project files are not replaced.
The checkout command selects the exact revision imported here.

```powershell
git clone --filter=blob:none --sparse https://github.com/xahidbuffon/SUIM.git SUIM-upstream
git -C SUIM-upstream sparse-checkout set models utils
git -C SUIM-upstream checkout 59f7cb0d4d13289434ad6e3ab162ab7a5dcc5c84
Copy-Item -LiteralPath .\SUIM-upstream\models -Destination . -Recurse
Copy-Item -LiteralPath .\SUIM-upstream\utils -Destination . -Recurse
Copy-Item -LiteralPath .\SUIM-upstream\LICENSE -Destination .\LICENSE-SUIM
Copy-Item -Path .\SUIM-upstream\train_*.py -Destination .
Copy-Item -Path .\SUIM-upstream\test_*.py -Destination .
Copy-Item -LiteralPath .\SUIM-upstream\get_f1_iou.py -Destination .
```

These commands copy the original upstream versions; they do not apply the local
output configuration above.

The temporary `SUIM-upstream/` checkout is retained for reference. The two
requested folders, license, and 13 entrypoint scripts are copied into the project root.

## Imported file checksums

These are the original upstream checksums before the local output configuration
edits above.

| Upstream path | Git blob SHA-1 | SHA-256 |
| --- | --- | --- |
| `LICENSE` | `0c10d7c45ef8855e670927d7c35ef7a6ad7fc6ca` | `ff4b621aaedcb8a62a1a4ae8446b9b90885baea94dc98d87ee801eb257bb0f36` |
| `models/BilinearUpSampling.py` | `ff4c206fcffa3fa88ea426e22604e763ac7b111c` | `e2bc80a4c5dc849002f3cadb7da0fe1db3e629c0b5356ece261d520fadd26d8f` |
| `models/__init__.py` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `models/basic_models.py` | `5c5556bfe6f0647312464a8eae2ccd68561c52cf` | `fe54886bdc775b6961a8ac85df84656d85019b80c752c8b3829208b359ef069b` |
| `models/deeplabv3.py` | `15087d64171761a94bb7d5217992e5db7dc08b7c` | `d42c991002abe340ac875589b41a778b17119a7dd355d6ed2b51c3743043c2d3` |
| `models/fcn.py` | `581e96bf8be2772096a4e0a2722821c200f69ca7` | `8a25b8e1afc04cbe9b165313c941f4e2f72a2c4d1463062a8893f9d0905492c4` |
| `models/mobilenet.py` | `ce3e88e54a9d6059716c6a72441c8452ef9b7cff` | `7db39d88f30f29c7cdc7f387cd1f735a9c2d866fb005be46147d0765b4ccd912` |
| `models/pspnet.py` | `c0c4c30382945bf954ee1e769b2248b05c15ee20` | `368b05c9b9c38b21356ab5aea1a8bf487ab99a80f4f8a10c670bd5ebf0519ebd` |
| `models/resnet50.py` | `30fe0eeff3eb48f4b163b26851ff88cf96c392ce` | `34460d3bc174aa46178e64ff078c7972703d519ef2fb1b60aac5ecb08977f479` |
| `models/segnet.py` | `8bd016c824aebb629fa434c4842f902260ac4b6d` | `e3843e3a32a15081ac128fb888433ba9dc1a8bab66398ca3c317b1e12ffc8fbc` |
| `models/suim_net.py` | `4f3484c9223c6d537e4b615d1fd684937fcd010b` | `729f3f7164a203f453ab88407f89c75366768a40308dc80d0bf1baf8cfda87eb` |
| `models/unet.py` | `ebaa1a54380ea1cc1d935e0a78dcb2d841e08318` | `c4cfedad9be1602e8fdce01b5a784916efa17e4fffd678ad2323617e2ca0bb64` |
| `models/vgg16.py` | `89a8a48238ec968916364b3be3c0e159fb77cfb9` | `a2b7e89399bc49d809dade6b0d103187502a297b37b0743336f59bcdbd942a12` |
| `utils/__init__.py` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `utils/data_utils.py` | `fd9d23b1242340c23cca95aace210742e467f6cc` | `2d0534b677aed10d5891d1ef170b99b2c1e6560902444f7821eb47ca110542b2` |
| `utils/measure_utils.py` | `4d3b7e180682f6d7066627a33c11332eaa4a486a` | `8b7de75dbe8069af6220501cab976b681bd74796f0eda4c11ea56b5aaa6322a0` |
| `get_f1_iou.py` | `5308afdd7c0ff326862a4dd9222fd3c3b0cdc625` | `1cd6c91686fdfd39015c5345071eeac5362af3493e4ee9db29e43b3a31a972ce` |
| `test_deeplab.py` | `98f64001257ce20424fdfd50e4b966cab764533a` | `ef55842f8319e1cfb60dcade7ce9bee0dc855c0ba199ca36a56387b5496dba62` |
| `test_fcn8.py` | `6870b7ff722352ca7105e7c2b8818fd947448964` | `cf2bf700a976b5f4770f83fe5df4a180d960ae3eba1b19256151dc1054609d1b` |
| `test_pspnet.py` | `10145b9c6e6a0fd33d71c5815e8c9ffb0b116f10` | `612f7257aae9b1fa07d9dc9e4df145450a82f71692de841d9e22f36bd4260e3a` |
| `test_segnet.py` | `8a1f4fcc606ebb67d7d64e37e2ad55f8e5376fd6` | `5015ccca2627db906768bd0863b1e4f62df2dd624d7c52d3cd8968f7d4616d27` |
| `test_suimnet.py` | `608c0c999f3b0032dbcd54d91b9404470dcd4cdf` | `c69ff09dc14f97cad861e2743e8fa2713da791e574abb25dff8668712cc73fa3` |
| `test_unet.py` | `c61825007f83c9a21772e0e75f070e8a23d5ed3e` | `76ba200f8bc82d41e6a8cef9f72f88f5f3f11da579d496cb13b088a0a0c24f26` |
| `train_deeplab.py` | `3328d20af7225aa0922b5f511da6e11c3b32c325` | `fc4fead808d2c2744db4519d48dcc1c1f77c586bc3f9eea4d4e5d555221882c5` |
| `train_fcn8.py` | `9941ef9eb31b2b7a8bdbd6786958e5d549b6e7ed` | `86f10dcda6cc948f67461012631d4af2693c89b29272f1c00ab086e69828e26e` |
| `train_pspnet.py` | `517d0c0f9e6cc6fc50450c3dce98c39a2d53d45a` | `f81605818f503719cf44da9d6db20b60f00b614b49b36d61c1fe52cc233bb4f4` |
| `train_segnet.py` | `db62a7bed57b688bcbd91aec7cb2b7272f77f112` | `d0c359655edce12cce7f97b6f907c9335b9f993d54c098be6a3fbb2adae38e6b` |
| `train_suimnet.py` | `3f5bb3511b3b885a696f23f5859cd6340101ae98` | `04e5315b444d19df1daf1fc1d396f4c67f08af29d5076d561fabf0a17cfb081f` |
| `train_unet.py` | `243540128301017877c618fb04591f030724da27` | `d6cf600ad77c2e06fb8b8f0ea673c8d26a4b915634497c4a38286544abc0acca` |

# Data format

The released repository does not include data. Training and evaluation scripts expect an image dataset on disk using the `torchvision.datasets.ImageFolder` layout:

```text
data/yxdata/
  train/0/*.png
  train/1/*.png
  val/0/*.png
  val/1/*.png
  test/0/*.png
  test/1/*.png
```

Class labels:

```text
0 = FRB-like astrophysical segment
1 = RFI or non-astrophysical segment
```

Each image is a dynamic-spectrum segment produced from the PSRFITS-to-image preprocessing described in the manuscript. The model-side loader converts images to grayscale, resizes them to `288 x 288`, applies the validation/test transform without augmentation, and applies a small random affine translation during training when configured.

SNR-stratified evaluation uses the SNR value encoded in each image file name. The helper function extracts SNR from the second-to-last underscore-delimited token in the file stem. For example:

```text
FRB20201124A_tracking-M01_0011_9.59461_16.0962.png
```

has `SNR = 9.59461`. The default split is:

```text
low SNR:  SNR < 12
high SNR: SNR >= 12
```

The same parsing rule is applied to all segments used in the SNR split.

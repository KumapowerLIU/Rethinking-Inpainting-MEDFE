# Data and structure targets

## Source mix

The modern checkpoint uses **1,497,344 training entries**: 296,145 Unsplash-derived photos,
901,730 Places365-Standard images, and 299,469 HDTF face entries. The latter represent
30,128 unique extracted frames, repeated 9 or 10 times. Total distinct source records:
**1,228,003**. Repetition is deliberate sampling weight, not additional unique images.

- **Unsplash:** obtain access through the [dataset site](https://unsplash.com/data) and
  [official schema](https://github.com/unsplash/datasets). Our existing TFRecord collection
  is a local selection; it is not identical to the downloadable public Lite dataset.
- **Places365-Standard:** start with the [CSAIL repository](https://github.com/CSAILVision/places365).
  Our 901,730 records are a fixed half of its 1,803,460-image training archive, not the whole
  Places Challenge set. Rebuilding a half with another seed gives a different selection.
- **HDTF:** use the [official data/preprocessing instructions](https://github.com/MRzzm/HDTF).
  Our face images are selected extracted video frames. Frames from the same video are correlated;
  create any independent face validation split at the video/identity level.

The main validation manifest contains 2,000 held-out Unsplash records. The archived manifests
have no overlapping source records. It does not provide an independent HDTF face benchmark.
Training data and private storage paths are not bundled with the source package. Access to the
source datasets alone does not reconstruct our exact local manifest selection.

## Ordinary images: the current protocol

Create the train/validation split before indexing. Keep masks and generated results outside
these folders, since the script indexes every supported image recursively.

```bash
python scripts/build_image_index.py --images-root /data/photos/train --manifest data/train.jsonl
python scripts/build_image_index.py --images-root /data/photos/val --manifest data/validation.jsonl
```

Each JSONL row is an absolute path:

```json
{"format":"image","path":"/data/photos/train/example.jpg"}
```

With `data.format=indexed_rtv`, processing is:

1. Correct EXIF orientation, decode RGB, and resize to the configured square with BICUBIC.
2. Solve RTV on that RGB image and cache the float32 structure target.
3. Apply the same training flip to RGB and structure; generate the hole mask.

The RTV parameters are `lambda=0.015`, `sigma=3`, `sharpness=0.001`, `iterations=3`.
This is a sparse-system RTV solver, not a Gaussian blur. Caches are content/protocol addressed;
repeated source images reuse targets. At 256², one float32 RGB structure is approximately
0.75 MiB. A full 1.23M-image cache is roughly 0.9 TiB before metadata; it grows as samples are read.
Use fast storage and account for initial CPU work.

The existing training reader also accepts TAR offsets:

```json
{"format":"tar","path":"/data/train.tar","offset":512,"length":17076,"name":"class/example.jpg"}
```

and the established TFRecord row `[absolute_path, record_offset, total_record_length]`.
The TFRecord decoder expects the existing bytes-only `state_dict/raw_image` schema;
it is not a generic decoder for every TFRecord dataset. Offsets must be generated from the
actual archive, not copied from the examples above.

The current mixed recipe enables `data.allow_repeated_records=true`. To recreate face
oversampling, repeat those records in the training manifest before training. The reader
rejects conflicting definitions of the same source record; validation rejects repetitions.

## Masks

`True` or white (`>127`) denotes a hole. `False` or black denotes known pixels.
The random generator chooses rectangles, brush strokes, or mixtures within 10%–40% area.
A rectangle is a valid random mask; random does not mean every sample must be a free-form stroke.
Training masks vary by seed/epoch/index. Validation uses fixed masks and transformations.

`center` is an explicit ablation. File masks are supported by the **paired** reader;
`indexed_rtv` supports random/center masks. RGB, structure and mask must share geometry.

## Paired RGB + existing structure targets

The older paired protocol computes RTV at the original image resolution before training
transforms. It is intentionally different from the modern resize-then-RTV protocol.

```bash
medfe prepare-data --images-root /data/train --cache-root data/rtv-pairs \
  --manifest data/train-paired.jsonl --workers 4
```

Paired rows use `id`, `image`, `structure`, and optionally `mask`; relative paths resolve
against the manifest directory. Switch to `data.format=paired` and set
`data.allow_repeated_records=false` when using them. See the historical configs for this path.

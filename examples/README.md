# Inference example

This 256x256 masked image and white-is-hole mask come from the fixed local Unsplash-derived validation selection. `expected.png` is the completion from the modern MEDFE step120,994 generator. No ground-truth content from inside the hole is needed for inference.

```bash
medfe infer --checkpoint checkpoints/medfe-sd-partial.pt \
  --image examples/masked.png --mask examples/mask.png \
  --output outputs/completed.png --device cuda:0
```

Known pixels must remain unchanged. Small generated-pixel differences may occur across GPU/software backends. The example is for checking the interface, not a benchmark score.

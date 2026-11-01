# MEDFE architecture illustration

Generated with the built-in GPT Image tool on 2026-11-01.

- Asset: [medfe-architecture.png](medfe-architecture.png)
- Visual reference: Figure 2, page 5 of the [ECCV 2020 MEDFE paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470715.pdf).
- Architecture depicted: the current `sd_partial` variant, not the original 2020 backbone.
- Source checks: `src/medfe/model.py`, `src/medfe/blocks.py`, `src/medfe/sd_backbone.py`.
- Input/output image tiles are schematic illustrations, not model evaluation samples.
- The skip paths are grouped for readability. The post-bottleneck feature is E6; refined E6 starts the decoder, and refined E5 through E1 are concatenated at the matching decoder resolutions.

## Initial generation prompt

Use case: infographic-diagram.
Asset type: polished academic architecture figure for a GitHub README, MEDFE image inpainting.

Create a NEW high-resolution landscape scientific diagram, approximately 2:1 aspect ratio. The provided image is Figure 2 of the author's original ECCV 2020 MEDFE paper: use its disciplined academic design language and network layout as the visual reference, but redraw beautifully and depict the MODERN architecture specified below. Do not copy the reference's old decoder or its exact feature routing if inconsistent with this specification.

Visual design: white background, razor-sharp dark-gray thin strokes, elegant restrained typography similar to Helvetica, consistent generous spacing, small flat axonometric tensor slabs like a carefully drawn CVPR/ECCV figure. Subtle light gray encoder slabs, muted blue decoder slabs, warm apricot PartialConv slabs, sage dashed skip connections, restrained terracotta feature-routing arrows. Flat fills with only geometric face shading. No gradients, glow, drop shadows, glossy 3D, decorative blobs, emojis, ornamental icons, fake photographic thumbnails or marketing badges. Scientific visual clarity is the priority. No giant title. Small title top left "MEDFE"; subtitle "Modern PartialConv / residual backbone".

COMPOSITION: wide U-shaped encoder-decoder feature pyramid across the upper half, with a clearly organized dual-branch MEDFE feature-fusion inset occupying the lower half. Similar to the reference, arrows should be orthogonal and carefully separated. Text must be legible at normal README width.

UPPER NETWORK:
At far left a simple flat image-grid card and binary-mask card, caption "Masked RGB + valid mask". Do not generate a photoreal scene. Feed into SIX progressively smaller encoder feature slabs, collectively titled "PartialConv residual encoder". Below this title a small subtitle "GroupNorm · SiLU · mask propagation". Label their spatial side lengths in order: "128", "64", "32", "16", "8", "4". Encoder flows from left to right down to a compact "Residual bottleneck" at the center. At right show SIX progressively larger blue decoder slabs, collectively titled "Residual decoder", with subtitle "Nearest upsampling + 3×3 Conv". Decoder feature side lengths in order: "8", "16", "32", "64", "128", "256". They lead to "RGB prediction", then a small "Copy known pixels" compositor, then a final flat RGB image-grid card captioned "Completed image". A very thin direct route from input to the compositor can run along the top margin, labelled "Known pixels".

SKIP AND FUSION SEMANTICS:
The six encoder tensors E1..E6, including the bottleneck as E6, remain available. Each is refined by ADDING a scale-specific projection of the equalized fusion feature. Refined E6 starts the decoder; refined E5..E1 are CONCATENATED with decoder features at the matching resolutions. Draw sage dashed skip paths for encoder features, plus circled "+" nodes for feature addition, with a shared terracotta equalization-return bus reaching these nodes. Route the resulting refined features into the decoder. Keep crossings minimal and avoid an accidental plain bypass that omits equalization. Label this routing compactly "Equalization-refined skips". If necessary abbreviate the individual routes as a clearly labelled multi-scale bus, rather than draw ambiguous tangled arrows. Correct topology matters more than extra wires.

LOWER FUSION INSET:
Use a subtle light-gray hairline bounding box titled "MEDFE mutual feature fusion" with a small "32 × 32 feature space" note.
Deep encoder features (16, 8, 4) go through align/concatenate into the TOP row labelled "Structure".
Shallow encoder features (128, 64, 32) go through align/concatenate into the BOTTOM row labelled "Texture".
Keep these two inputs separate. Each row has its own orange multi-scale filling group showing THREE PARALLEL horizontal streams, labelled "3×3", "5×5", "7×7". Indicate stacks of partial convolution blocks, with ellipsis representing repeated blocks. A small shared caption reads "5 PartialConv layers per stream". The structure and texture streams SHARE WEIGHTS between branches; show this with a discreet bracket or note "Shared weights". Do NOT depict structure and texture as sequential stages.
Each three-stream group merges into its own filled-feature stack. Those TWO filled-feature stacks concatenate and project into one "Fuse" node, then go through two consecutive clean rectangular modules labelled "Channel equalization" and "Spatial equalization / BPA". The resulting "Equalized features" tensor is routed back UP to supplement all six encoder scales via the addition nodes. This is a supplementary feature-fusion module, not the sole sequential encoder-decoder bottleneck.

A compact legend along the bottom: an apricot slab "PartialConv", a dashed sage arrow "Encoder skip", a circled plus "Addition". Optionally include the exact compact formula "Refined E_i = E_i + P_i(F_eq)" to disambiguate the addition; P_i is scale projection. No GAN, no losses, no attention transformer, no timestep, no noise vector, no diffusion, no pretrained SD weights in the drawing. Keep auxiliary training heads out of this overview. All labels must use exact clean English spelling. Do not include paper caption text or a fake figure number.

## Routing correction prompt

Edit the provided MEDFE diagram. Preserve its clean paper-like visual style, typography, color palette, upper feature pyramids and the entire lower dual-branch fusion inset. The lower fusion inset is very good. Make the following targeted connectivity correction in the horizontal band BETWEEN the encoder/decoder pyramids and the lower fusion inset.

The existing band is mathematically wrong: it has many plus symbols with no outputs, and arrows incorrectly coming DOWN from decoder features. DELETE ALL of those dangling plus nodes, ALL arrows from decoder slabs going down, the long peach "Equalization-refined skips" rectangle, and all dangling green lines. Replace that band with a clean, explicitly ABSTRACTED multi-scale routing diagram:
1. A small left box "Encoder features E₁ … E₆". A single sage collector/bracket BELOW the six gray encoder slabs and the residual bottleneck represents the set of encoder features. Connect that collector to this box. E₆ denotes the post-bottleneck feature. Keep the encoder's normal sequential forward chain.
2. Center box "Refine each scale", containing a second line exactly "E′ᵢ = Eᵢ + Pᵢ(F_eq)". The left encoder-feature box enters this center box from the left.
3. ONE terracotta route from the lower inset's purple equalized feature tensor F_eq goes UP into the center refinement box. It must have a clear arrowhead entering that box.
4. The center box outputs to a right box "Refined feature pyramid".
5. This right box has TWO outgoing connections: a solid arrow labelled "4×4: decoder start" going UP into the FIRST blue decoder stage (the one outputting side length 8), and a thin sage dashed arrow labelled "8–128: concatenate skips" pointing to a single bracket that covers the other five decoder stages. This latter bracket explicitly abstracts per-scale concatenation; do not draw six individual spaghetti routes. The legend should label sage dashed as "Multi-scale feature path", not imply anything adds to decoder outputs.
6. REMOVE the direct ordinary arrow from residual bottleneck to first blue decoder stage. The only start of the blue decoder chain comes from the refined feature pyramid, because the real network refines the bottleneck too.
Keep these routing boxes readable and visually light; enlarge the vertical gap a little if needed. No arrow should point OUT of a blue decoder slab toward a plus. There should be NO standalone plus symbols anywhere outside the small legend; addition is conveyed by the explicit equation inside the center box.

Additional exact correction: under "Masked RGB + valid mask", the binary validity-mask card must have BLACK in the hole and WHITE in known/background pixels. The current card is inverted. Invert only the mask card's colors. Preserve the masked RGB illustration and output illustration. Change the bottom legend dashed green caption to "Multi-scale feature path". You may remove the now redundant equation from the bottom legend because it is in the refinement box.

Do NOT change the lower Structure/Texture kernel streams, their weights sharing, channel equalization or BPA order. Keep spelling perfect. Keep output wide, all text inside canvas, no gradients or decorative effects. This is a scientific diagram: the corrected explicit routing is essential.

## Final correction prompt

Make a surgical correction to this existing MEDFE illustration. Keep the entire artistic layout, typography, palette, input/output cards, and the lower Structure/Texture streams unchanged. Do NOT redesign it.

Fix only these FIVE specific graph details:
(1) Restore SIX blue decoder slabs across the top right, labelled left-to-right "8", "16", "32", "64", "128", "256". There are only five now because "16" is missing. Make them slightly narrower if necessary so all six fit. Sequential arrows connect these six slabs left-to-right.
(2) The green encoder-feature collector must include the output of the "Residual bottleneck", because E6 is post-bottleneck. Extend the green collector/bracket rightward under this block, and add a short downward line from the block into that collector. Its other existing connection to "Encoder features E1 … E6" stays.
(3) Erase BOTH existing orange return paths from the LOWER fusion inset to the UPPER network: currently one incorrectly starts at the gray concatenated tensor, and one incorrectly terminates at the right-hand refined pyramid box. Draw exactly ONE new return path. It MUST START at the PURPLE tensor labelled F_eq (the last tensor, AFTER Spatial equalization / BPA), travel upward and left along the open space between the fusion inset and the routing boxes, and end with an UPWARD ARROWHEAD at the BOTTOM CENTER of the peach "Refine each scale" box. Do not start this line at the gray concatenate tensor. Do not connect F_eq directly to "Refined feature pyramid". There is exactly one equalization return line: purple F_eq -> peach Refine each scale.
(4) The only input to the blue "Refined feature pyramid" BOX is the black horizontal arrow from the peach refinement box. From the TOP of this blue BOX, draw the orange decoder-start arrow upward into the FIRST blue decoder SLAB, which is labelled "8". This start arrow must originate from the blue BOX, not from the horizontal wire before that box. Label the start arrow exactly "4x4: decoder start" using an ASCII x; existing "4X4"/"4A4" text must be clean. Retain the dashed green "8–128: concatenate skips" arrow from this blue BOX toward the bracket covering the other FIVE blue slabs.
(5) In the bottom fusion inset, label the gray tensor AFTER the circled C and BEFORE Channel equalization "Fuse" instead of "Concatenate", since that tensor includes channel projection; circled C still denotes concatenate.

Do not remove any encoder scale. Do not change the lower Structure, Texture, three parallel kernel streams, channel and spatial equalization blocks, or the legend. No new arrows other than the ones specified. Perfectly typeset English. All edges must have deliberate endpoints, with no ambiguous floating wires.

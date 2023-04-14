# GraphCut Segmentation Improver (gcsi)

DISCLAIMER: written by another author, interpreting the implementation.

This program tries to improve segmentation masks produced by e.g. atlas voting by using the original MRI image's intensities (or derivatives of this - such as classification maps) to guide where the boundaries for the segments should be.

This is done by taking an intensity image I, a mask M, and a list of probabilities maps P (could be produced by e.g. atlas voting), one for each label L that is wanted in the output. These probability maps along with their labels are considered the input segmentation that the program tries to improve.

In addition foreground and background probability maps are supplied to guide the algorithm on what intensities belong to different classes / labels.

See e.g. https://se.mathworks.com/help/images/segment-image-using-graph-cut.html for a visual interpretation of how this could be understood.

Typically input for foreground and background probability maps in a segmentation pipeline could be the WM, GM, and CSF probability maps produced by a classification algorithm. The classification probability maps for WM, GM, and CSF can then be input as foreground and background in different combinations depending on what label L we are trying to improve.

The program then for each label L and its probability map P finds the optimal solution to a energy function with a number of different terms. Each of these terms can be weighted by constants tuning the energy function optimization for different purposes. These terms can be changed by specifying new values when invoking the program.

The energy function is optimized using the more general max-flow/min-cut GraphCut algorithm, which generally can be applied to image processing problems.

Voxel/Pixel Labeling as a Graph Cut problem, is excellent explained in section 4 of [Collins, Toby: Graph Cut Matching In Computer Vision](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0405/COLLINS/TobyCollinsAVAssign2.pdf)

The implementation is based on the article [Wolz: Segmentation of subcortical structures and the hippocampus in brain MRI using graph-cuts and subject-specific a-priori information](https://ieeexplore.ieee.org/document/5193086)

Possible extensions and/or deviations from the original article:

* Normalizing the input image
* Narrow band optimization (reduced 40 percent runtime and 60 percent memory footprint)
* Boundary term from gradient magnitude image (ContourPropability)

## Usage

It can be used as both GraphCutCleaner and as CSFGraphCutSegmenter by setting different foreground and background inputs. When using more than one foreground and/or background image the combined probability is dire mean value of each voxel in the input. If another combined probability is wanted, then it should be combined prior to using this application.

### Improving hippocampus segmentation (GraphCutCleaner)

```bash
gcsi bias_corrected.nii atlas_masks_raw.nii output_labels.nii \
-c 17 -f left_hippocampus_atlas_probability_map.nii \
-c 53 -f right_hippocampus_atlas_probability_map.nii \
-f gm_classification_probability_map.nii.gz \
-b wm_classification_probability_map.nii.gz \
-b csf_classification_probability_map.nii.gz \
-l 1.0 -C 0.1 -a 0.1 -t 0.001 -n 0 -I 299 -U 512 -d 255 -B 32
```

### Improving classification of CSF (CSFGraphCutSegmenter)

```bash
gcsi bias_corrected_image.nii.gz icv_mask.nii.gz output_graphcut_csf_mask.nii \
-c 1 -p csf_atlas_probability_map.nii.gz \
-f csf_classification_probability_map.nii.gz \
-b wm_classification_probability_map.nii.gz \
-b gm_classification_probability_map.nii.gz \
-l 1.0 -C 0.1 -a 0.25 -t 0 -n 0.001 -I 0 -U 255 -d 255 -B 8
```

### Improving the brain mask (BrainGraphCutSegmenter)

Note: have only been used experimentally to try to improve the removal of CSF from brain masks.

Same as GraphCutCleaner but with the following changes:

* fC: 0.1 -> 0.02
* fAlpha: 0.1 -> 0.21
* fTissueBackgroundProbability:
  0.5f*(fWMProbability + fCSFProbability) -> fCSFProbability
* fTissueForegroundProbability:
  fGMProbability -> 0.5f*(fWMProbability + fGMProbability)

Also have the changes:

* UpperThreshold: 512 -> 255
* InsideValue: 299 -> 0

Which are the same values as in CSFGraphCutSegmenter, these values where however also set to this in an older version of the GraphCutCleaner program, so maybe it just has not been updated!

```bash
gcsi bias_corrected.nii icv_mask.nii csf_cleaned_brainmask.nii \
-c 1 -f brain_probability_map.nii \
-f gm_classification_probability_map.nii.gz \
-b wm_classification_probability_map.nii.gz \
-b csf_classification_probability_map.nii.gz \
-l 1.0 -C 0.02 -a 0.21 -t 0.001 -n 0 -I 0 -U 255 -d 255 -B 32
```

Default parameters not changed have been taken from current version of GraphCutCleaner, which might not be correct as GraphCutCleaner has been change since BrainGraphCutSegmenter was in use.

## Further references

* [Lotjonen: Fast and robust multi-atlas segmentation of brain magnetic resonance images](https://www.sciencedirect.com/science/article/abs/pii/S1053811909010970)
* [Boykov: Graph Cuts and Efficient N-D Image Segmentation](https://link.springer.com/article/10.1007/s11263-006-7934-5)
* [Zhuang Song, Nicholas Tustison, Brian Avants, and James C. Gee: Integrated Graph Cuts for Brain MRI Segmentation](https://link.springer.com/chapter/10.1007/11866763_102)

## TODOs

* remove GraphCutSegmentationImprover forward decl
* search for TODO in code
* fix so that running with different number of threads gives same output

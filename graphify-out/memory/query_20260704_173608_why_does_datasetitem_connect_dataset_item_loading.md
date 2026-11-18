---
type: "query"
date: "2026-07-04T17:36:08.932820+00:00"
question: "Why does DatasetItem connect Dataset Item Loading to 14 other communities?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["DatasetItem", "BaseDataset", "ProcessingPipeline"]
---

# Q: Why does DatasetItem connect Dataset Item Loading to 14 other communities?

## Answer

Expanded from original query via vocab: [dataset, item, pipeline, camera, intrinsic, visualization, metric, group, stereo, kitti, nyu, vkitti]. DatasetItem (datasets/base.py L12, degree 65) is the single currency of the evaluation pipeline: every dataset loader's find_items() produces DatasetItem objects, every load_gt_depth()/load_image()/load_intrinsic() consumes them, and ProcessingPipeline (_process_item_group, _generate_comparison_outputs, _is_already_processed, _load_existing_metrics in src/pipeline.py) groups them by item_id/camera_id for stereo comparison. It bridges 14 communities because producers (KITTI/NYU/VKITTI/DIODE/Cityscapes/DrivingStereo/Middlebury loaders), consumers (pipeline, visualization, metrics), and verifiers (test_data_loading, test_integration) each cluster separately but all pass this one dataclass. All 53 EXTRACTED edges are imports/references; the 13 INFERRED 'uses' edges on it are correct — they mirror the extracted find_items references.

## Outcome

- Signal: useful

## Source Nodes

- DatasetItem
- BaseDataset
- ProcessingPipeline
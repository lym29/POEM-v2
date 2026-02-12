# POEM-v2 Codebase Documentation

**POEM-v2** (Point-Embedded Multi-view Transformer v2) is a generalizable multi-view hand mesh recovery model designed for real-world hand motion capture and teleoperation.

## Overview

POEM-v2 is a PyTorch-based deep learning framework that reconstructs 3D hand meshes from multi-view camera inputs. It's designed to be flexible, robust, and production-ready.

### Key Features

- **Flexible Multi-view**: Works with any number and arrangement of cameras
- **Robust to Occlusion**: Handles partial visibility and occlusions
- **Absolute Position**: Produces hand position in real-world (meter) units
- **Dual Hand Support**: Supports both left and right hands
- **Published Work**: TPAMI (v2) and CVPR (v1)

### Publications

- **POEM-v2**: "Generalizable Multi-view Hand Mesh Recovery with Flexible Camera Arrangement" (TPAMI)
- **POEM-v1**: "Point-Embedded Multi-view Stereo for Hand Mesh Recovery" (CVPR)

## Project Structure

```
third_party/POEM-v2/
├── lib/                          # Main library code
│   ├── datasets/                 # Dataset implementations
│   ├── models/                   # Model architectures
│   ├── utils/                    # Utility functions
│   ├── metrics/                  # Evaluation metrics
│   ├── viztools/                 # Visualization tools
│   ├── fit/                      # Fitting utilities
│   ├── external/                 # External packages (CMR, METRO)
│   └── data_wds/                 # WebDataset utilities
├── scripts/                      # Training and evaluation scripts
├── config/                       # Configuration files (YAML)
├── tool/                         # Inference tools
├── transform/                    # Transform utilities
├── prepare/                      # Data preparation scripts
├── video_tool/                   # Video processing tools
├── docs/                         # Documentation
└── assets/                       # Asset files (MANO model, etc.)
```

## Core Components

### 1. Models (`lib/models/`)

#### Main Model: POEM (`POEM.py`)

**Class**: `PtEmbedMultiviewStereoV2` (714 lines)

The core Point-Embedded Multi-view Transformer model.

**Key Features**:
- Multi-view feature extraction with shared backbone
- Point-embedded transformer for 3D reasoning
- Supports both parametric (MANO) and non-parametric outputs
- Flexible camera arrangement handling

**Architecture**:
```
Input: Multi-view images + camera parameters
  ↓
Backbone (ResNet/HRNet) → Per-view features
  ↓
Point-Embedded Transformer → 3D hand representation
  ↓
Prediction Head → MANO parameters or 3D vertices
  ↓
Output: 3D hand mesh (778 vertices, 21 joints)
```

**Important Methods**:
- `forward()`: Main forward pass
- `_prepare_input()`: Prepare multi-view inputs
- `_extract_features()`: Extract per-view features
- `_aggregate_multiview()`: Aggregate multi-view information

#### Base Models

- **PETR.py** (`PETRMultiView`): Point-based Estimation of Transformer
  - Base transformer-based multi-view model
  - Used as foundation for POEM

- **MVP.py**: MVP model variant (inherits from PETR)

#### Backbones (`lib/models/backbones/`)

- **HRNet** (`hrnet.py`): High-Resolution Network
  - Maintains high-resolution representations
  - Best for detailed hand features
  - Pretrained on ImageNet

- **ResNet** (`resnet.py`): Residual Network
  - Standard backbone option
  - Faster but less detailed

- **Hourglass** (`hourglass.py`): Stacked Hourglass Network
  - Multi-scale feature extraction

#### Prediction Heads (`lib/models/heads/`)

- **ptEmb_head.py**: Point-Embedded head (49KB)
  - Main prediction head for POEM
  - Outputs MANO parameters or vertices

- **petr_head.py**: PETR head
- **petr_FTL_head.py**: PETR with Feature-level Temporal Learning
- **mvp_head.py**: MVP head

#### Building Blocks (`lib/models/bricks/`)

- **transformer.py**: Standard transformer blocks
- **point_transformers.py**: Point-based transformers
- **metro_transformer.py**: METRO-style transformers
- **conv.py**: Convolution blocks

#### Layers (`lib/models/layers/`)

- **mano_wrapper.py**: MANO hand model wrapper
  - Interfaces with MANO parametric model
  - Handles shape and pose parameters

- **ptEmb_transformer.py**: Point-embedded transformer layers
- **petr_transformer.py**: PETR transformer layers
- **mvp_decoder.py**: MVP decoder

### 2. Datasets (`lib/datasets/`)

#### Base Dataset Class (`hdata.py`)

**Class**: `HandDataset`

Base class for all hand datasets with common functionality:
- Multi-view image loading
- Camera parameter handling
- Data augmentation
- MANO parameter loading

#### Supported Datasets

1. **HO3D** (`ho3d.py`, `ho3d_official_test.py`)
   - Hand-Object 3D interaction dataset
   - 10 test sequences
   - Multi-view RGB images with depth
   - MANO annotations

2. **DexYCB** (`dexycb.py`)
   - Dexterous grasping of YCB objects
   - 8 synchronized camera views
   - MANO parameters and object poses

3. **Arctic** (`arctic.py`)
   - Bimanual hand-object manipulation
   - Articulated objects
   - Both hands annotated

4. **InterHand** (`interhand.py`)
   - Large-scale interacting hands dataset
   - 2.6M frames
   - Multi-view setup

5. **OakInk** (`oakink.py`, `oakink2_dev.py`)
   - Diverse object categories
   - Hand-object contact annotations
   - OakInk v2 in development

6. **FreiHAND** (`freihand.py`)
   - Single-hand dataset
   - RGB images with 3D annotations

7. **YT3D** (`yt3d.py`)
   - YouTube 3D hand dataset

#### Mixed Dataset (`mix_dataset.py`)

**Class**: `MixWebDataset`

Combines multiple datasets for training:
- Configurable mixing ratios
- Per-dataset sampling strategies
- WebDataset format for efficiency

**Configuration Example**:
```yaml
DATASET:
  TRAIN:
    TYPE: MixWebDataset
    DATASET_LIST: [HO3D, DexYCB, Arctic, Interhand, Oakink, Freihand]
    MIX_RATIO: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    EPOCH_SIZE: 210000
```

#### WebDataset Support (`lib/data_wds/`)

- **multiview_wds.py**: Multi-view WebDataset implementation
- **dumper.py**: Dataset dumping utilities
- Efficient streaming data loading
- Supports large-scale datasets

### 3. Configuration System (`lib/utils/config.py`)

Uses **YACS** (Yet Another Configuration System) with custom extensions.

#### Config Structure

```yaml
TRAIN:
  BATCH_SIZE: 8
  EPOCH: 10
  OPTIMIZER: adam
  LR: 0.001
  SCHEDULER: StepLR
  GRAD_CLIP: 1.0

DATASET:
  TRAIN:
    TYPE: MixWebDataset
    DATASET_LIST: [...]
  VAL:
    TYPE: HO3D
    SPLIT: test

MODEL:
  TYPE: PtEmbedMultiviewStereoV2
  BACKBONE:
    TYPE: HRNet
    PRETRAINED: path/to/pretrained.pth
  HEAD:
    TYPE: ptEmb_head
    NUM_QUERIES: 778  # MANO vertices

DATA_PRESET:
  IMAGE_SIZE: [256, 256]
  HEATMAP_SIZE: [64, 64]
  BBOX_EXPAND_RATIO: 1.5
```

#### Config Files (`config/`)

- **backbone/**: Backbone-specific configs
  - `cls_hrnet_w40_*.yaml`: HRNet-W40 config
  - `cls_hrnet_w64_*.yaml`: HRNet-W64 config

- **release/**: Release model configs
  - `train_small.yaml`: Small model (fast)
  - `train_medium.yaml`: Medium model (balanced)
  - `train_large.yaml`: Large model (accurate)
  - `train_huge.yaml`: Huge model (best quality)
  - `train_medium_MANO.yaml`: Medium with MANO output
  - `eval_single.yaml`: Single-view evaluation

#### Custom Config Node (`CN`)

**Class**: `CN` (extends YACS CfgNode)

**Key Features**:
- Recursive config merging
- Command-line argument integration
- Type checking and validation

**Usage**:
```python
from lib.utils.config import CN, get_config

# Load config
cfg = get_config()
cfg.merge_from_file("config/release/train_medium.yaml")
cfg.merge_from_list(["TRAIN.BATCH_SIZE", "16"])
cfg.freeze()
```

### 4. Training Pipeline (`scripts/`)

#### Distributed Training (`train_dp.py`, `train_dp_wds.py`)

**Main Script**: `train_dp_wds.py` (with WebDataset support)

**Features**:
- Multi-GPU training with DistributedDataParallel
- Automatic mixed precision (AMP)
- Gradient clipping
- Learning rate scheduling
- Checkpoint management
- TensorBoard logging

**Training Loop**:
```python
1. Load config and setup distributed environment
2. Build model, optimizer, scheduler
3. Load datasets (WebDataset for efficiency)
4. Training loop:
   - Forward pass
   - Compute losses
   - Backward pass with gradient clipping
   - Optimizer step
   - Log metrics
5. Validation and checkpointing
```

**Command**:
```bash
python scripts/train_dp_wds.py \
    --cfg config/release/train_medium.yaml \
    --exp_id my_experiment \
    TRAIN.BATCH_SIZE 16
```

#### Key Training Parameters

- `TRAIN.BATCH_SIZE`: Batch size per GPU
- `TRAIN.EPOCH`: Number of epochs
- `TRAIN.LR`: Learning rate
- `TRAIN.OPTIMIZER`: Optimizer type (adam, adamw, sgd)
- `TRAIN.SCHEDULER`: LR scheduler (StepLR, CosineAnnealingLR)
- `TRAIN.GRAD_CLIP`: Gradient clipping threshold

### 5. Evaluation (`scripts/`)

#### General Evaluation (`eval.py`)

Evaluates model on test datasets.

**Metrics**:
- **MPJPE**: Mean Per Joint Position Error (mm)
- **PA-MPJPE**: Procrustes-Aligned MPJPE (mm)
- **MPVPE**: Mean Per Vertex Position Error (mm)
- **PA-MPVPE**: Procrustes-Aligned MPVPE (mm)
- **AUC**: Area Under Curve
- **PCK**: Percentage of Correct Keypoints

**Command**:
```bash
python scripts/eval.py \
    --cfg config/release/eval_single.yaml \
    --checkpoint path/to/checkpoint.pth \
    --dataset HO3D
```

#### Single-View Evaluation (`eval_single.py`)

Evaluates single-view performance.

#### HO3D Official Evaluation (`eval_ho3d_official.py`)

Runs official HO3D benchmark evaluation.

### 6. Utilities (`lib/utils/`)

#### Transform Utilities (`transform.py`, 47KB)

**Key Functions**:

**Camera Transformations**:
- `cam2pixel()`: 3D camera coords → 2D pixel coords
- `pixel2cam()`: 2D pixel coords → 3D camera coords
- `world2cam()`: World coords → camera coords
- `cam2world()`: Camera coords → world coords

**Rotation Representations**:
- `axis_angle_to_matrix()`: Axis-angle → rotation matrix
- `matrix_to_axis_angle()`: Rotation matrix → axis-angle
- `quaternion_to_matrix()`: Quaternion → rotation matrix
- `rotation_6d_to_matrix()`: 6D rotation → rotation matrix
- `euler_angles_to_matrix()`: Euler angles → rotation matrix

**Heatmap Processing**:
- `generate_heatmap()`: Generate 2D heatmaps from keypoints
- `heatmap_to_coord()`: Extract coordinates from heatmaps
- `soft_argmax()`: Differentiable argmax for heatmaps

**Batch Operations**:
- All functions support batched inputs
- GPU-accelerated with PyTorch

#### Builder Pattern (`builder.py`, 11KB)

**Registry System** for modular component building.

**Usage**:
```python
from lib.utils.builder import DATASET, MODEL, BACKBONE

# Register components
@DATASET.register_module
class MyDataset:
    pass

@MODEL.register_module
class MyModel:
    pass

# Build from config
dataset = DATASET.build(cfg.DATASET)
model = MODEL.build(cfg.MODEL)
```

**Registries**:
- `DATASET`: Dataset registry
- `MODEL`: Model registry
- `BACKBONE`: Backbone registry
- `HEAD`: Head registry
- `LOSS`: Loss function registry

#### Testing Utilities (`testing.py`, 16KB)

**Callbacks and utilities for evaluation**:
- Metric computation
- Result saving
- Visualization generation

#### Triangulation (`triangulation.py`)

**3D triangulation from multi-view 2D projections**.

**Key Functions**:
- `triangulate_point_from_multiple_views()`: DLT triangulation
- `triangulate_ransac()`: RANSAC-based robust triangulation

#### Distributed Training (`dist_utils.py`)

**Utilities for distributed training**:
- Process group initialization
- Gradient synchronization
- Distributed data loading

#### I/O Utilities (`io_utils.py`)

**File I/O operations**:
- JSON/pickle loading and saving
- Image loading and saving
- Mesh I/O

#### Logging (`logger.py`, `recorder.py`, `summary_writer.py`)

**Experiment tracking and logging**:
- Console logging with colors
- TensorBoard integration
- Experiment recording

### 7. Metrics (`lib/metrics/`)

#### Basic Metrics (`basic_metric.py`)

- **MPJPE**: Mean Per Joint Position Error
- **MPVPE**: Mean Per Vertex Position Error

#### Procrustes Alignment (`pa_eval.py`)

- **PA-MPJPE**: Procrustes-Aligned MPJPE
- **PA-MPVPE**: Procrustes-Aligned MPVPE
- Removes global rotation and translation

#### Percentage of Correct Keypoints (`pck.py`)

- **PCK**: Percentage within threshold
- **AUC**: Area under PCK curve

#### Mean End-Point Error (`mean_epe.py`)

- **EPE**: End-Point Error for 2D keypoints

### 8. Visualization (`lib/viztools/`)

#### Drawing Utilities (`draw.py`, 21KB)

**Functions**:
- `draw_hand_mesh()`: Render hand mesh on image
- `draw_skeleton()`: Draw hand skeleton
- `draw_keypoints()`: Draw 2D/3D keypoints
- `draw_bbox()`: Draw bounding boxes

**Renderers**:
- OpenCV-based rendering (fast)
- OpenDR rendering (high quality)
- Open3D rendering (interactive)

#### OpenDR Renderer (`opendr_renderer.py`)

High-quality mesh rendering with lighting.

#### Open3D Utilities (`viz_o3d_utils.py`)

Interactive 3D visualization with Open3D.

### 9. Fitting Utilities (`lib/fit/`)

#### Hand Loss (`hand_loss.py`)

**Loss functions for hand fitting**:
- Joint position loss
- Vertex position loss
- Shape regularization
- Pose regularization

#### Silhouette Loss (`silhouette_loss.py`)

**Silhouette-based loss** for hand-object fitting.

#### PyTorch3D Renderer (`pytorch3d_renderer.py`)

Differentiable rendering with PyTorch3D.

#### Frame Fitting (`fit/frame_fit/`)

Per-frame hand fitting utilities.

### 10. External Packages (`lib/external/`)

#### CMR (`cmr/`)

Collaborative Multi-view Reconstruction model.

#### METRO (`metro/`)

METRO (Mesh Transformer) model with HRNet backbone.

## Data Preparation

### Dataset Directory Structure

Multi-view datasets are located on remote cluster `hanhai22-01` at `~/DATA/POEM-v2/`:

```
~/DATA/POEM-v2/
├── HO3D_mv/              # HO3D multi-view training
├── HO3D_mv_test/         # HO3D multi-view test
├── DexYCB_mv/            # DexYCB multi-view
├── Arctic_mv/            # Arctic multi-view
├── Interhand_mv/         # InterHand multi-view
├── Oakink_mv/            # OakInk multi-view
└── Freihand_mv/          # FreiHAND multi-view
```

### Data Format

Each dataset follows a common structure:

```
{dataset}_mv/
├── images/               # Multi-view images
│   ├── cam0/
│   ├── cam1/
│   └── ...
├── annotations/          # MANO parameters, 3D keypoints
├── camera_params/        # Intrinsics and extrinsics
└── metadata.json         # Dataset metadata
```

### WebDataset Format

For efficient training, datasets are converted to WebDataset format:

```
{dataset}_wds/
├── train-000000.tar
├── train-000001.tar
└── ...
```

Each tar file contains:
- `{id}.jpg`: Image
- `{id}.json`: Annotations
- `{id}.cam.json`: Camera parameters

## Inference

### Hand Inference Tool (`tool/infer_hand.py`, 12KB)

**Standalone inference script** for hand mesh recovery.

**Usage**:
```python
from tool.infer_hand import HandInference

# Initialize
inferencer = HandInference(
    config_path="config/release/eval_single.yaml",
    checkpoint_path="path/to/checkpoint.pth",
    device="cuda:0"
)

# Infer from multi-view images
images = [...]  # List of images
camera_params = [...]  # List of camera parameters

result = inferencer.infer(images, camera_params)
# result contains:
#   - vertices: (778, 3) hand mesh vertices
#   - joints: (21, 3) hand joints
#   - mano_params: MANO shape and pose parameters
```

**Features**:
- Batch inference
- Multi-view or single-view
- MANO parameter output
- Visualization support

### Flip Utility (`tool/flip_util.py`)

**Hand flipping utilities** for left/right hand conversion.

## Key Algorithms

### 1. Point-Embedded Transformer

**Core innovation of POEM-v2**.

**Concept**:
- Embed 3D hand points (vertices/joints) as queries
- Use transformer to aggregate multi-view features
- Each query attends to relevant image regions across views

**Advantages**:
- Flexible camera arrangement
- Robust to occlusion
- Learns 3D structure implicitly

### 2. Multi-View Feature Aggregation

**Process**:
1. Extract per-view features with shared backbone
2. Project 3D queries to each view
3. Sample features at projected locations
4. Aggregate with attention mechanism
5. Decode to 3D hand mesh

### 3. MANO Parametric Model

**MANO** (hand Model with Articulated and Non-rigid defOrmations):
- Shape parameters: 10D (hand shape)
- Pose parameters: 48D (16 joints × 3D rotation)
- Output: 778 vertices, 21 joints

**Integration**:
- Model predicts MANO parameters
- MANO layer generates mesh
- Differentiable for end-to-end training

## Training Strategies

### 1. Mixed Dataset Training

Train on multiple datasets simultaneously:
- Increases diversity
- Improves generalization
- Configurable mixing ratios

### 2. Multi-Stage Training

1. **Stage 1**: Train backbone on 2D keypoint detection
2. **Stage 2**: Train full model with frozen backbone
3. **Stage 3**: Fine-tune end-to-end

### 3. Data Augmentation

- Random rotation, scaling, translation
- Color jittering
- Random occlusion
- View dropout (for multi-view robustness)

### 4. Loss Functions

**Supervised Losses**:
- Joint position loss (L1/L2)
- Vertex position loss (L1/L2)
- MANO parameter loss

**Regularization**:
- Shape regularization (prevent extreme shapes)
- Pose regularization (prevent unnatural poses)

**Optional**:
- Silhouette loss (for hand-object fitting)
- Contact loss (for hand-object interaction)

## Common Workflows

### Training a New Model

```bash
# 1. Prepare config
cp config/release/train_medium.yaml config/my_experiment.yaml
# Edit config as needed

# 2. Train
python scripts/train_dp_wds.py \
    --cfg config/my_experiment.yaml \
    --exp_id my_experiment \
    TRAIN.BATCH_SIZE 16

# 3. Monitor with TensorBoard
tensorboard --logdir exp/my_experiment/log
```

### Evaluating a Model

```bash
# Evaluate on HO3D
python scripts/eval.py \
    --cfg config/release/eval_single.yaml \
    --checkpoint exp/my_experiment/checkpoint_best.pth \
    --dataset HO3D

# Evaluate on multiple datasets
for dataset in HO3D DexYCB Arctic; do
    python scripts/eval.py \
        --cfg config/release/eval_single.yaml \
        --checkpoint exp/my_experiment/checkpoint_best.pth \
        --dataset $dataset
done
```

### Running Inference

```python
from tool.infer_hand import HandInference

# Initialize
inferencer = HandInference(
    config_path="config/release/eval_single.yaml",
    checkpoint_path="exp/my_experiment/checkpoint_best.pth"
)

# Load images and camera parameters
images = [...]  # List of PIL Images or numpy arrays
cameras = [...]  # List of camera dicts with 'K', 'R', 't'

# Infer
result = inferencer.infer(images, cameras)

# Visualize
from lib.viztools.draw import draw_hand_mesh
vis_img = draw_hand_mesh(images[0], result['vertices'], cameras[0])
```

### Visualizing Dataset

```bash
# Visualize multi-view dataset
python scripts/viz_multiview_dataset.py \
    --dataset HO3D \
    --data_root ~/DATA/POEM-v2/HO3D_mv_test \
    --sequence GPMF10 \
    --frame 0
```

## Important Files Reference

### Configuration
- `lib/utils/config.py`: Config system implementation
- `config/release/train_medium.yaml`: Standard training config (236 lines)

### Models
- `lib/models/POEM.py`: Main POEM model (714 lines)
- `lib/models/PETR.py`: PETR base model (21KB)
- `lib/models/layers/mano_wrapper.py`: MANO integration

### Datasets
- `lib/datasets/hdata.py`: Base dataset class
- `lib/datasets/ho3d.py`: HO3D dataset (40KB)
- `lib/datasets/mix_dataset.py`: Mixed dataset loader

### Training
- `scripts/train_dp_wds.py`: Main training script
- `scripts/eval.py`: Evaluation script

### Utilities
- `lib/utils/transform.py`: Transform utilities (47KB)
- `lib/utils/builder.py`: Registry system (11KB)
- `lib/utils/testing.py`: Testing utilities (16KB)

### Visualization
- `lib/viztools/draw.py`: Drawing utilities (21KB)
- `scripts/viz_multiview_dataset.py`: Dataset visualization

### Inference
- `tool/infer_hand.py`: Inference tool (12KB)

## Dependencies

### Core Dependencies
- PyTorch >= 1.10
- torchvision
- numpy
- opencv-python
- pillow

### 3D and Visualization
- pytorch3d
- open3d
- trimesh
- pyrender

### Data Processing
- webdataset
- h5py
- scipy

### Utilities
- yacs (configuration)
- tqdm (progress bars)
- tensorboard (logging)
- matplotlib

### MANO Model
- Requires MANO model files in `assets/mano/`
- Download from https://mano.is.tue.mpg.de/

## Tips and Best Practices

### Training
- Start with `train_medium.yaml` for balanced performance
- Use WebDataset format for large-scale training
- Monitor validation metrics to prevent overfitting
- Use gradient clipping to stabilize training
- Adjust batch size based on GPU memory

### Evaluation
- Always evaluate on official test splits
- Report both MPJPE and PA-MPJPE
- Use consistent image preprocessing
- Check camera parameter correctness

### Debugging
- Use `viz_multiview_dataset.py` to verify data loading
- Enable verbose logging with `--verbose`
- Visualize predictions with `draw_hand_mesh()`
- Check intermediate features with hooks

### Performance Optimization
- Use mixed precision training (AMP)
- Increase num_workers for data loading
- Use WebDataset for I/O efficiency
- Profile with PyTorch profiler

## Common Issues

### 1. MANO Model Not Found
**Error**: `FileNotFoundError: MANO model not found`
**Solution**: Download MANO model and place in `assets/mano/`

### 2. Camera Parameter Mismatch
**Error**: Incorrect 3D reconstruction
**Solution**: Verify camera intrinsics (K) and extrinsics (R, t) are correct

### 3. Out of Memory
**Error**: `CUDA out of memory`
**Solution**: Reduce batch size or image resolution

### 4. WebDataset Loading Slow
**Error**: Slow data loading
**Solution**: Increase `num_workers` or use local SSD for data

### 5. Multi-view Inconsistency
**Error**: Inconsistent predictions across views
**Solution**: Check camera synchronization and calibration

## Integration with Main Project

POEM-v2 is integrated into the main project for multi-view hand pose evaluation:

### Usage in Main Project

```python
# In eval_hand/multiview/eval_multiview.py
from third_party.POEM-v2.lib.datasets import HO3DDataset
from third_party.POEM-v2.lib.models import PtEmbedMultiviewStereoV2

# Load POEM-v2 dataset
dataset = HO3DDataset(
    data_root="~/DATA/POEM-v2/HO3D_mv_test",
    split="test",
    **config
)

# Use POEM-v2 config format
from third_party.POEM-v2.lib.utils.config import CN
config = CN()
config.IMAGE_SIZE = [256, 256]
config.TRANSFORM = None  # No augmentation for eval
```

### Dataset Wrappers

The main project provides wrappers in `eval_hand/multiview/`:
- `ho3d_multiview.py`: HO3D wrapper
- `dexycb_multiview.py`: DexYCB wrapper
- `arctic_multiview.py`: Arctic wrapper
- `interhand_multiview.py`: InterHand wrapper
- `oakink_multiview.py`: OakInk wrapper

These wrappers adapt POEM-v2 datasets to the main project's evaluation framework.

## Future Work

### Planned Improvements
- Real-time inference optimization
- Support for more datasets
- Improved occlusion handling
- Temporal consistency for video
- Hand-object interaction modeling

### Research Directions
- Self-supervised learning
- Few-shot adaptation
- Domain adaptation
- Weakly-supervised training

## References

### Papers
- POEM-v2: "Generalizable Multi-view Hand Mesh Recovery with Flexible Camera Arrangement" (TPAMI)
- POEM-v1: "Point-Embedded Multi-view Stereo for Hand Mesh Recovery" (CVPR)
- MANO: "Embodied Hands: Modeling and Capturing Hands and Bodies Together" (SIGGRAPH Asia 2017)

### Related Projects
- MANO: https://mano.is.tue.mpg.de/
- HO3D: https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/
- DexYCB: https://dex-ycb.github.io/
- Arctic: https://arctic.is.tue.mpg.de/

## Contact and Support

For questions about POEM-v2:
- Check the original repository documentation
- Review published papers for methodology details
- Examine example configs in `config/release/`

For integration with the main project:
- See `eval_hand/multiview/` for usage examples
- Check `CLAUDE.md` in the main project root

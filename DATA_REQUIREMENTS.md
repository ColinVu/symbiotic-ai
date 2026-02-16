# Data Requirements for Symbiote Pipeline

## Overview

This document lists ALL data files, configuration files, and setup requirements needed before running the symbiote pipeline. Use this as a checklist to ensure your system is properly configured.

**Last Updated**: February 15, 2026

---

## Required Before ANY Pipeline Use

### 1. ARUCO Marker Configuration (`config/aruco_bins.json`)

**Location**: `symbiotic-ai/config/aruco_bins.json`

**Purpose**: Maps ARUCO marker IDs to pick/place bins and objects

**Required Fields**:
```json
{
  "marker_dict": "DICT_4X4_1000",
  "bins": {
    "MARKER_ID": {
      "type": "pick" or "place",
      "object": "object_name",
      "description": "human readable description"
    }
  },
  "distance_decay": 5.0
}
```

**Example**:
```json
{
  "marker_dict": "DICT_4X4_1000",
  "bins": {
    "0": {
      "type": "pick",
      "object": "apple",
      "description": "Red apple pick bin"
    },
    "1": {
      "type": "pick",
      "object": "banana",
      "description": "Yellow banana pick bin"
    },
    "10": {
      "type": "place",
      "object": "apple",
      "description": "Red apple place bin"
    },
    "11": {
      "type": "place",
      "object": "banana",
      "description": "Yellow banana place bin"
    }
  },
  "distance_decay": 5.0
}
```

**How to Create**:
1. Decide on ARUCO dictionary (default DICT_4X4_1000 for 3-digit IDs 0-999)
2. Assign unique IDs to each physical bin
3. Map IDs to "pick" or "place" type
4. Specify which object each bin contains
5. Print markers and affix to physical bins
6. Create JSON file with mappings

**Validation**: Run ARUCO test tool (see section below)

---

## Required for Basic Training (Image-Based)

### 2. Image Dataset

**Location**: User-defined (e.g., `symbiotic-ai/images/training/`)

**Structure**:
```
images/training/
├── apple/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── banana/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── orange/
    ├── image001.jpg
    └── ...
```

**Requirements**:
- One folder per object class
- Folder name = object label
- Images must show hand holding object
- Minimum: 20 images per class
- Recommended: 50-100 images per class
- Supported formats: JPG, PNG, HEIC

**Already in Pipeline**: 
- Hand segmentation (automatic)
- Blur filtering (automatic)
- CLIP embedding (automatic)

---

## Required for Video Training (Basic, No HMM)

### 3. Training Videos

**Location**: User-defined (e.g., `symbiotic-ai/videos/training/`)

**Requirements**:
- Clear view of hand and objects
- ARUCO markers visible in frame when near bins
- Hand must be segmentable by MediaPipe
- Minimum resolution: 720p
- Recommended: 1080p, 30 FPS
- Format: MP4, AVI, MOV

**What You Provide**:
- Videos showing pick/place operations
- No annotation needed for basic training
- ARUCO config must match physical setup

**CLI Usage**:
```bash
python -m symbiote.cli.main train \
    --video videos/training/pick_apple_001.mp4 \
    --label "apple" \
    --aruco-config config/aruco_bins.json
```

---

## Required for HTK HMM State Detection Training

### 4. Annotated Training Videos

**Location**: Same as training videos

**Structure**:
```
videos/training/
├── pick_place_001.mp4
├── pick_place_001_annotations.csv    # ← REQUIRED
├── pick_place_002.mp4
├── pick_place_002_annotations.csv    # ← REQUIRED
└── ...
```

**Annotation CSV Format**:

**File**: `video_name_annotations.csv`

**Columns**: `timestamp_start`, `timestamp_end`, `state`

**Example**:
```csv
timestamp_start,timestamp_end,state
0.0,1.5,CARRY_EMPTY
1.5,3.2,PICK
3.2,8.7,CARRY_WITH
8.7,10.5,PLACE
10.5,15.0,CARRY_EMPTY
15.0,16.8,PICK
16.8,23.1,CARRY_WITH
23.1,25.0,PLACE
25.0,30.0,CARRY_EMPTY
```

**Rules**:
- States MUST follow cycle: PICK → CARRY_WITH → PLACE → CARRY_EMPTY
- No state skipping allowed
- Timestamps in seconds (float)
- No gaps between consecutive states
- States must cover entire video duration

**How to Create Annotations**:

**Option 1: Manual annotation**
1. Watch video frame-by-frame
2. Note timestamps where state changes occur
3. Create CSV with transitions
4. Validate state cycle is correct

**Option 2: Video annotation tools**
- Use tools like CVAT, Label Studio, or VGG Video Annotator
- Export to CSV format matching above schema

**Minimum Data Volume**:
- 5-10 annotated videos
- 3-5 complete pick-place cycles per video
- Total: 15-50 state cycles

**Recommended Data Volume**:
- 10-20 annotated videos
- 5-10 cycles per video
- Total: 50-200 state cycles
- Diverse conditions (different objects, lighting, speeds)

---

## Required for HTK HMM Training

### 5. HTK Toolkit Installation

**What**: Hidden Markov Model Toolkit from Cambridge University

**Download**: http://htk.eng.cam.ac.uk/

**License**: Free for research use (HTK License required)

**Installation**:
1. Register and download HTK source code
2. Compile for your platform (Linux/Mac/Windows)
3. Add HTK binaries to PATH
4. Required executables: `HCompV`, `HERest`, `HVite`, `HHEd`

**Verify Installation**:
```bash
HCompV -V  # Should print HTK version
HERest -V  # Should print HTK version
```

**Alternative**: Pre-compiled binaries (if available for your platform)

---

## Physical Setup Requirements

### 6. ARUCO Marker Printing

**What**: Physical ARUCO markers for bin identification

**Steps**:
1. Generate markers from DICT_4X4_1000 dictionary (IDs 0-999)
   - Use online generator: https://chev.me/arucogen/
   - Or use OpenCV Python script
2. Print markers (recommended size: 6-8 inches square)
3. Laminate or mount on rigid backing
4. Affix to bins in clear view
5. Ensure good lighting (avoid glare)

**Example Python Generator**:
```python
import cv2
import numpy as np

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)

for marker_id in [0, 1, 2, 10, 11, 12]:
    marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, 400)
    cv2.imwrite(f"aruco_marker_{marker_id}.png", marker_image)
```

### 7. Physical Bin Setup

**Requirements**:
- Separate bins for "pick" and "place" operations
- Each bin labeled with unique ARUCO marker
- Bins arranged so camera can see markers
- Consistent bin positions during training
- One object type per bin

**Recommended Setup**:
```
[PICK BINS]          [WORK AREA]          [PLACE BINS]
  ARUCO 0                                    ARUCO 10
  (Apple)                                    (Apple)
  
  ARUCO 1                                    ARUCO 11
  (Banana)            [CAMERA]               (Banana)
  
  ARUCO 2                                    ARUCO 12
  (Orange)                                   (Orange)
```

---

## Optional Configuration Files

### 8. Custom CLIP Model (Optional)

**Default**: Uses `openai/clip-vit-base-patch32` from Hugging Face

**Custom Model Location**: User-defined

**When to Use Custom**:
- Training on specialized objects not in CLIP's training data
- Need higher accuracy for specific domain
- Already have fine-tuned CLIP model

**How to Specify**:
```python
# In symbiote/core/config.py
MODEL = "path/to/your/custom/clip/model"
```

### 9. Training Configuration (Optional)

**Default**: Uses `DEFAULT_CONFIG` in `symbiote/core/config.py`

**Customizable Parameters**:
```python
{
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "hidden_dim": 256,
    "dropout": 0.3,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42
}
```

**Override via CLI**:
```bash
python -m symbiote.cli.main train \
    --video video.mp4 \
    --label "apple" \
    --lr 0.0005 \
    --epochs 50 \
    --hidden-dim 512
```

---

## Directory Structure Checklist

Before running the pipeline, ensure this structure exists:

```
symbiotic-ai/
├── config/
│   └── aruco_bins.json                 # ✓ REQUIRED
│
├── videos/
│   ├── training/
│   │   ├── video_001.mp4
│   │   ├── video_001_annotations.csv   # For HMM training only
│   │   └── ...
│   └── testing/
│       └── ...
│
├── images/
│   └── training/
│       ├── apple/
│       ├── banana/
│       └── ...
│
├── models/
│   └── classifier/
│       ├── .cache/                     # Auto-created
│       └── htk_models/                 # Auto-created for HMM
│
└── symbiote/
    └── ...  # Code
```

---

## Validation Checklist

Before training, verify:

### Configuration
- [ ] `config/aruco_bins.json` exists
- [ ] All physical ARUCO IDs are in config
- [ ] Marker types (pick/place) are correct
- [ ] Object names match training data labels

### Physical Setup
- [ ] ARUCO markers printed and affixed to bins
- [ ] Markers are clearly visible
- [ ] Good lighting (no glare on markers)
- [ ] Camera can see markers and hand simultaneously

### Data
- [ ] Training videos/images exist
- [ ] Videos show clear hand movements
- [ ] For HMM: Annotation CSVs exist and match video names
- [ ] For HMM: Annotations follow state cycle rules

### Software
- [ ] Python environment has all dependencies
- [ ] For HMM: HTK toolkit installed and in PATH
- [ ] For HMM: Can run `HCompV -V` successfully

### Test Tools
- [ ] Can run ARUCO test tool:
```bash
python -m symbiote.state_detection.test_aruco_detection \
    --video test_video.mp4 \
    --output test_annotated.mp4 \
    --aruco-config config/aruco_bins.json
```
- [ ] Annotated video shows correct marker detection
- [ ] Weighted scores look reasonable

---

## Quick Start Workflows

### Workflow 1: Image-Based Training (Simplest)
```bash
# 1. Setup (one-time)
# - Create config/aruco_bins.json
# - Print and affix ARUCO markers
# - Collect training images in folders by class

# 2. Train
python -m symbiote.cli.main train \
    --image-dir images/training \
    --aruco-config config/aruco_bins.json
```

**Required Data**:
- ✓ ARUCO config
- ✓ Training images
- ✗ Training videos
- ✗ Annotations
- ✗ HTK toolkit

### Workflow 2: Video-Based Training (No HMM)
```bash
# 1. Setup (one-time)
# - Same as Workflow 1
# - Record training videos

# 2. Train
python -m symbiote.cli.main train \
    --video videos/training/pick_apple.mp4 \
    --label "apple" \
    --aruco-config config/aruco_bins.json
```

**Required Data**:
- ✓ ARUCO config
- ✓ Training videos
- ✗ Training images (optional)
- ✗ Annotations
- ✗ HTK toolkit

### Workflow 3: Full HMM State Detection Training
```bash
# 1. Setup (one-time)
# - Same as Workflow 2
# - Annotate videos with state timestamps
# - Install HTK toolkit

# 2. Test ARUCO detection
python -m symbiote.state_detection.test_aruco_detection \
    --video videos/training/pick_place_001.mp4 \
    --output test_annotated.mp4 \
    --aruco-config config/aruco_bins.json

# 3. Train HMM
python -m symbiote.cli.main train \
    --video videos/training/pick_place_001.mp4 \
    --label "apple" \
    --train-hmm \
    --annotation videos/training/pick_place_001_annotations.csv \
    --aruco-config config/aruco_bins.json
```

**Required Data**:
- ✓ ARUCO config
- ✓ Training videos
- ✓ Annotations (CSV files)
- ✓ HTK toolkit installed
- ✗ Training images (optional)

---

## Common Issues and Solutions

### Issue: "Cannot find aruco_bins.json"
**Solution**: Create `config/aruco_bins.json` with proper structure (see section 1)

### Issue: "No ARUCO markers detected"
**Solution**: 
- Check lighting (avoid glare)
- Verify markers are printed from correct dictionary (DICT_4X4_1000 for IDs 0-999)
- Ensure markers are in camera view
- Run ARUCO test tool to validate

### Issue: "Hand detection failed"
**Solution**:
- Ensure hand is clearly visible
- Check lighting conditions
- Verify MediaPipe can detect hand (21 landmarks)

### Issue: "State cycle validation failed"
**Solution**:
- Check annotation CSV follows: PICK → CARRY_WITH → PLACE → CARRY_EMPTY
- Ensure no state skipping
- Verify timestamps are monotonically increasing

### Issue: "HTK command not found"
**Solution**:
- Verify HTK installation: `which HCompV`
- Add HTK bin directory to PATH
- Recompile HTK if needed

---

## Data Storage Estimates

### Disk Space Requirements

**Per Training Video** (30 seconds, 1080p):
- Original video: ~50-100 MB
- CLIP embedding cache: ~2-4 MB (for extracted frames)
- HTK feature cache: ~15 KB (very compact)
- Annotation CSV: <1 KB

**Per Training Image** (1920x1080):
- Original image: ~2-5 MB
- CLIP embedding cache: ~2-4 KB

**Trained Models**:
- CLIP classifier: ~10-50 MB
- HTK HMM models: ~500 KB - 1 MB (very compact)

**Total for Complete Setup**:
- 20 training videos: ~2-3 GB
- 500 training images: ~2-3 GB
- All caches and models: ~500 MB
- **Total: ~5-7 GB**

---

## Summary: What You MUST Have

### Minimum to Run Anything:
1. ✓ `config/aruco_bins.json`
2. ✓ Physical ARUCO markers (printed and affixed)

### To Train Classifier (Image):
3. ✓ Training images organized by class

### To Train Classifier (Video):
4. ✓ Training videos showing pick/place operations

### To Train HMM State Detection:
5. ✓ Annotated training videos (CSV files)
6. ✓ HTK toolkit installed

### To Test/Validate:
7. ✓ Test videos
8. ✓ ARUCO test tool working

---

**Next Steps**: 
1. Use this document as a checklist
2. Create required data files
3. Validate with test tools
4. Begin training pipeline

For implementation details, see:
- `HTK_STATE_DETECTION_IMPLEMENTATION.md` - Full HTK system design
- `QUICK_START_NEW_FEATURES.md` - Usage examples
- `README_REFACTORED.md` - Pipeline overview

# Model Card: HemoScan YOLO Blood Cell Detector

## Model
- Architecture: YOLO11n
- Task: Object detection of Platelets, RBC, WBC in blood smear images
- Input: RGB image, 640x640
- Output: bounding boxes + class labels + confidence

## Training
- Dataset: blood smear dataset (765 train / 73 val / 36 test)
- Epochs: 100 (early stopping)
- Image size: 640
- Optimizer: Ultralytics default

## Performance (test set, conf=0.50)
- Macro Precision: 0.739
- Macro Recall: 0.805
- Macro F1: 0.745
- mAP@50: 0.833
- mAP@50:95: 0.577

## Limitations
- Small test set (36 images)
- Class imbalance
- Over-detection of RBCs
- Not clinically validated

## Intended use
Research and educational use only. Not for diagnosis.
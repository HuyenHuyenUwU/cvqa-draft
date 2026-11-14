# Data pipeline

```mermaid
flowchart LR
    RAW[FracAtlas / BTXRD] --> ANN[annotations.py]
    ANN --> QA[qa_generation.py<br/>weak_template]
    ANN --> SPLIT[splits.py<br/>hash split, patient-level nếu có patient_id]
    QA --> DS[BaseCVQADataset]
    SPLIT --> DS
    DS --> PRE[DicomPreprocessor<br/>reader→windowing→normalization→transforms]
    PRE --> COL[collate.py<br/>has_box / has_answer]
    COL --> DL[DataLoader]
```

- QA sinh từ template được gắn nhãn `template_weak_supervision` (không phải ground truth chuyên gia).
- Không có bbox → `has_box=False` (ghi rõ "missing", không bịa).
- Không tự tải dữ liệu: đặt đường dẫn trong `configs/data.yaml`.

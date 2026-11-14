# Kiến trúc hệ thống

Offline và Online **dùng chung** `BoneCVQASystem` (`src/bone_cvqa/system.py`). Khác biệt chỉ là chế độ chạy và các thành phần riêng từng giai đoạn.

```mermaid
flowchart LR
    IMG[Ảnh DICOM/PNG] --> IP[image_processing]
    Q[Câu hỏi + lịch sử] --> QP[question_processing]
    IP --> ONT[ontology_processing<br/>placeholder]
    QP --> ONT
    IP --> CMF[cross_modal_fusion<br/>cross-attention + LLM decoder]
    QP --> CMF
    ONT --> CMF
    CMF --> AG[answer_generation]
    subgraph OFFLINE_ONLY[Chỉ Offline]
      JL[training/joint_loss] --> OPT[optimizer + backprop]
    end
    subgraph ONLINE_ONLY[Chỉ Online]
      IQ[image_quality] --> IP
      SD[scope_detector] --> QP
      UE[uncertainty_estimation<br/>placeholder] --> TV[trustworthy_verification<br/>warning + final output]
    end
    AG --> JL
    AG --> UE
    AG --> TV
```

| Thành phần | Offline | Online |
|---|---|---|
| image_processing (DICOM, vision encoder, localization*, RoI, global) | ✔ | ✔ |
| question_processing (language encoder, entity linking*, type prompt) | ✔ | ✔ |
| ontology_processing* | ✔ | ✔ |
| cross_modal_fusion + LLM decoder, answer_generation | ✔ | ✔ |
| image_quality, scope_detector | ✘ | ✔ |
| uncertainty_estimation*, trustworthy_verification | ✘ | ✔ |
| joint_loss, optimizer, backprop, checkpoint save | ✔ | ✘ |

`*` = placeholder, ghi log `[TODO] ... is a placeholder`.

## Gradient
`commons/execution_mode.py::execution_context(module, mode)` là điểm duy nhất quyết định:
TRAIN → `module.train()` + `enable_grad`; INFER → `module.eval()` + `inference_mode`.

## Model ngoài (ablation)
`external_models/` + mục `external_models` trong `configs/model.yaml`. Mỗi model implement `BaseModelAdapter` (`load/predict/unload`), đăng ký theo `(role, provider)`.

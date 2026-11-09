# Bone Disease CVQA — Codebase mẫu (Offline / Online)

Codebase module hóa cho hệ thống **Conversational Visual Question Answering (CVQA)** trên ảnh X-quang xương, bám theo hai sơ đồ `pipeline_offline.png` và `pipeline_online.png`.

> **Trạng thái trung thực:** toàn bộ pipeline đã chạy end-to-end trên CPU với **mô hình ngẫu nhiên cỡ nhỏ** (cùng lớp `open_clip.TimmModel`, `HFTextEncoder`, `Qwen2ForCausalLM`) và dữ liệu tổng hợp. **Chưa** chạy với trọng số BiomedCLIP / Qwen thật và **chưa** chạy trên FracAtlas / BTXRD thật (môi trường tạo code không có các tệp này và không tự tải). Xem mục 22.

## 1. Tổng quan & 2. Mục tiêu nghiên cứu
Xây dựng mô hình hội thoại trực quan tin cậy trên ảnh X-quang xương, kết hợp: biểu diễn ảnh, hiểu câu hỏi theo ngữ cảnh hội thoại, Bone Disease Ontology, hợp nhất đa phương thức, sinh câu trả lời, ước lượng bất định và kiểm chứng tin cậy. Giai đoạn này chỉ làm **baseline CVQA chạy được**; ontology / uncertainty là placeholder có interface sạch.

**Quyết định kiến trúc đã áp dụng:** dùng **text encoder có sẵn của BiomedCLIP** (BERT khởi tạo từ PubMedBERT) thay cho PubMedBERT riêng → ảnh và câu hỏi cùng không gian CLIP, bớt một checkpoint. Backend `hf` (PubMedBERT độc lập) vẫn có trong `ConversationTextEncoder` làm phương án dự phòng.

## 3. Kiến trúc Offline
`Input → 1. Image Processing → 2. Question Processing → 3. Ontology Processing (placeholder) → 4. Cross-modal Fusion → 5. Answer Generation → 6. Multi-task Learning (Joint Loss, Backprop & Optimizer)`

## 4. Kiến trúc Online
`DICOM Preprocessing → Image Quality Assessment → Vision Encoder → (Localization / ROI / Global)`; `Language Encoder → Entity Linking → Question Scope Detector → (in-scope) Type-Conditional Prompt`; ontology (placeholder) → Fusion → LLM → `Answer / Box / Evidence Graph / Explanation / Uncertainty / Confidence → Warning Generator → Final Output`. Câu hỏi `out_of_scope` bỏ qua sinh câu trả lời và đi thẳng tới Warning Generator.

Online **không import** `src.offline` (có test kiểm tra: `test_online_does_not_import_offline_trainer`). Checkpoint, cấu hình tiền xử lý và trọng số được nạp qua `src/models/model_io.py` dùng chung.

## 5. Bảng ánh xạ sơ đồ → mã nguồn

| Module (sơ đồ) | Vị trí | Trạng thái | Input | Output |
|---|---|---|---|---|
| DICOM Preprocessing | `data/dicom_preprocessing.py`, `image_loader.py`, `image_preprocessing.py` | implemented | file DICOM/PNG/JPG | tensor `[3,S,S]` |
| Image Quality Assessment (Online) | `online/image_quality.py` | implemented (heuristic) | gray min-max, kích thước gốc | label/score/warnings |
| Vision Encoder (BiomedCLIP) | `models/visual/biomedclip_encoder.py` | implemented | ảnh | `patch [B,N,Dv]`, `global [B,Dg]` |
| Lesion Localization | `models/localization/lesion_localizer.py` | implemented (cần nhãn box) / blocked_by_missing_data nếu thiếu | patch features | box xyxy chuẩn hóa + heatmap logits |
| ROI Features | cùng file (`RoIFeatureExtractor`) | implemented | patch, box, mask hợp lệ | `[B,Dv]` + `roi_valid` |
| Global Visual Embedding | `biomedclip_encoder.py` | implemented | ảnh | `[B,Dg]` |
| Conversation-aware Language Encoder | `models/language/text_encoder.py`, `data/text_processing.py` | implemented | câu hỏi + lịch sử | token `[B,T,Dt]`, embedding `[B,De]` |
| Entity Linking (Question) | `question_processing/entity_linking.py` | placeholder | text | `{"entities": [], "status": "not_implemented"}` |
| Question Scope Detector (Online) | `question_processing/question_scope_detector.py` | implemented (rule-based) | câu hỏi, entity, lịch sử | in/out_of_scope + reason |
| Type-Conditional Prompt | `question_processing/type_conditional_prompt.py` | implemented (rule-based) | câu hỏi, entity | question_type, type_id, prompt_text |
| Ontology Retriever | `ontology/retriever.py` | placeholder | ROI, global, entity, index | subgraph rỗng |
| Entity Linking (Ontology) | `ontology/entity_linker.py` | placeholder | ứng viên | entities rỗng |
| Knowledge Embedding | `ontology/knowledge_embedding.py` | placeholder | subgraph, entities | `embedding=None` |
| Cross-modal Fusion | `models/fusion/multimodal_fusion.py`, `projection.py` | implemented | global, ROI, question, type, knowledge(mask) | tokens `[B,K,d]` + mask |
| LLM adapter | `models/fusion/llm_adapter.py` | implemented | fused tokens | soft tokens `[B,n,H_llm]` |
| Medical LLM Decoder | `models/llm/llm_decoder.py`, `llm_loader.py` | implemented | soft tokens + prompt | hidden states / văn bản |
| Answer Head | `models/llm/answer_head.py` | implemented | hidden, labels | logits tại vị trí đáp án |
| Heatmap / Bounding Box | `online/output_formatter.py` | implemented / `not_available` nếu localizer chưa train | loc output | box/heatmap hoặc trạng thái |
| Evidence Graph | `ontology/evidence_graph.py` | placeholder | — | nodes/edges rỗng |
| Visual + Knowledge Explanation | `ontology/evidence_graph.py` | placeholder | — | `text=null` |
| Uncertainty Estimation | `uncertainty/estimator.py` | placeholder | logits, repr, answer | `score=null` |
| Confidence Score | `uncertainty/estimator.py` | placeholder | uncertainty | `score=null` |
| Joint Loss | `offline/losses.py` | implemented một phần: `L_ans`, `L_loc` active; `L_rat/L_kg/L_align` skipped | outputs + GT | total, active, skipped |
| Backprop & Optimizer | `offline/trainer.py`, `optimizer.py`, `checkpoint.py` | implemented | loss | checkpoint |
| Warning Generator | `online/warning_generator.py` | implemented (rule-based) | quality, scope, uncertainty, confidence | danh sách warnings |
| Final Output | `online/output_formatter.py` | implemented | tất cả | JSON schema thống nhất |
| xAI độc lập | — | not_applicable | | |

## 6–7. Module đã implement / placeholder
- **Placeholder** luôn trả `status: not_implemented`, ghi log một lần: `[TODO] <Module> is a placeholder. It will be implemented in a later development stage.`; **không** sinh embedding ngẫu nhiên, triple, độ tin cậy giả. Fusion che (mask) ô knowledge thay vì nạp giá trị bịa.
- **Rule-based (không phải mô hình học):** Type-Conditional Prompt (từ khóa), Question Scope Detector (từ khóa + lịch sử), Image Quality (heuristic, **không** phải đánh giá lâm sàng).
- **Fusion:** mỗi modality → Linear+LayerNorm → token + modality embedding → Transformer encoder nhỏ (thay được bằng cross-attention bằng cách thay `self.mixer`). **Adapter:** learned queries cross-attend → Linear sang `hidden_size` của LLM → soft tokens nối trước prompt qua `inputs_embeds`. LLM mặc định **đóng băng**; gradient vẫn chảy xuyên qua LLM về adapter/fusion.
- **Localization:** nhánh đơn giản kiểu LobA (saliency theo patch + hồi quy box), giám sát chỉ bằng box (L1 + GIoU + BCE heatmap theo mask từ box; không phải segmentation). Chỉ coi là "trained" sau `min_loc_steps` bước có nhãn.

## 8–10. Dữ liệu, cấu trúc thư mục, schema, giới hạn
Không tải dữ liệu tự động. Chạy trước: `python scripts/inspect_dataset.py --root /đường/dẫn --out outputs/inspect.json`.

**FracAtlas** (giả định theo tài liệu công khai — *cần xác minh*): `images/Fractured/*.jpg`, `images/Non_fractured/*.jpg`, `Annotations/COCO JSON/COCO_fracture_masks.json` (bbox `[x,y,w,h]`). Không có patient id → chia theo `image_id`.
**BTXRD**: schema **không giả định**; điền `columns`, `metadata_csv`, `annotation_dir` trong `configs/data/btxrd.yaml` sau khi inspect (xuất `.xlsx` → `.csv`). Có cột patient id thì chia theo bệnh nhân.

**Giới hạn QA (quan trọng):** hai bộ dữ liệu *không được giả định* có QA hội thoại chuyên gia. Mặc định `qa.source: weak_template` sinh câu hỏi–đáp mẫu từ nhãn/box, gắn `answer_source = template_weak_supervision`; **không** được báo cáo là ground truth chuyên gia, **không** có rationale. Có QA chuyên gia thì dùng `qa.source: file` (JSONL: `image_id, question, answer, turn`) → gắn `expert_annotation`. Hai loại không bao giờ trộn ngầm.
Chia tập: deterministic theo hash(seed, group key), chống rò rỉ bệnh nhân, ghi `split_metadata.json`; hỗ trợ file split chính thức.
Box được lưu chuẩn hóa theo ảnh gốc (xyxy, union mọi box); resize vuông làm méo tỉ lệ nhưng box chuẩn hóa vẫn đúng.

## 11–14. Cài đặt & cấu hình
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch: chọn bản CPU/CUDA phù hợp theo pytorch.org
```
Checkpoint (chỉ tải khi bạn đồng ý): đặt `biomedclip.local_dir`, `llm.local_dir` hoặc `allow_download: true` trong YAML. BiomedCLIP: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (open_clip). LLM: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0 theo model card — kiểm tra lại trước khi phân phối). Ước tính (chưa đo): ViT-B + BERT-base đóng băng + Qwen 1.5B fp16 ≈ 4–6 GB VRAM với batch 2 và `grad_accum_steps: 8`; dùng `load_in_4bit` nếu thiếu.
Cấu hình: `configs/offline.yaml` (include `data/*.yaml`, `model/*.yaml`), `configs/online.yaml`. Ghi đè: `--set train.epochs=3 data.root=/path`. Config cuối được lưu cạnh checkpoint (`config.yaml`).

## 15–17. Lệnh chạy
```bash
python scripts/prepare_dataset.py --config configs/offline.yaml --out outputs/prepared
python scripts/train.py    --config configs/offline.yaml --set data.root=/path/FracAtlas biomedclip.local_dir=/path/biomedclip llm.local_dir=/path/qwen
python scripts/evaluate.py --config configs/offline.yaml --checkpoint checkpoints/exp01/best.pt --split test
python scripts/infer.py    --config configs/online.yaml  --checkpoint checkpoints/exp01/best.pt \
       --image xray.jpg --question "Is there a fracture?" --history history.json --out result.json
```
Checkpoint chỉ lưu tham số `requires_grad` (fusion, adapter, localizer, phần encoder được mở khóa) + optimizer, scheduler, epoch, best metric, model/train config, split info, active losses; tháp đóng băng được nạp lại từ checkpoint gốc.
Đánh giá: Exact Match, token-F1 (ROUGE/BLEU ghi `not_computed`), IoU / hit-rate@{0.3,0.5} khi có box; ontology/uncertainty = `not_implemented`. Kết quả lưu JSON/CSV.

## 18–19. Ví dụ
Đầu vào `history.json`: `[{"question": "Is there a fracture?", "answer": "Yes, a fracture is present."}]`.
Đầu ra thật từ smoke test (mô hình ngẫu nhiên nên `answer` vô nghĩa) nằm ở `outputs/smoke/example_output.json`; các khóa: `sample_id, answer, localization, evidence_graph, visual_knowledge_explanation, uncertainty, confidence, image_quality, question_scope, warnings, pipeline_status, disclaimer, debug`. `uncertainty.score` và `confidence.score` luôn `null` cho tới khi có phương pháp được kiểm định; cảnh báo `low_confidence` chỉ phát khi `status == available` và có số thật.

## 20–21. Test
```bash
python -m pytest tests -q                 # 47 test, CPU, không tải checkpoint
python scripts/smoke_test.py              # train → eval → infer end-to-end với dữ liệu/mô hình tổng hợp
```

## 22. Giới hạn đã biết
- Chưa kiểm chứng với trọng số BiomedCLIP / Qwen thật và dữ liệu FracAtlas / BTXRD thật; layout FracAtlas là giả định, BTXRD chưa có schema.
- Trích patch feature dựa trên `TimmModel` của open_clip (đã kiểm chứng trên bản mã nguồn cài trong môi trường); checkpoint thật cần chạy `pytest tests/test_encoders.py` thay mô hình tiny để xác nhận shape.
- QA mặc định là weak supervision mẫu → số liệu answer chỉ phản ánh khả năng học template, không phải năng lực y khoa.
- Localization một box/ảnh (union); `Dv` của patch (768) khác `Dg` (512) và được project riêng.
- Scope detector có thể từ chối nhầm; Image Quality là heuristic chưa tinh chỉnh ngưỡng.
- Chưa có DDP/đa GPU; AMP chỉ trên CUDA.

## 23. Lộ trình
1. Ontology: dựng index Bone Disease Ontology → Retriever (ROI/global/entity → subgraph) → Entity Linking → Knowledge Embedding (GNN/KGE) → bật `knowledge_dim` trong fusion và `L_kg`.
2. Evidence Graph + Visual/Knowledge Explanation → `L_rat`, `L_align` (cần nhãn rationale).
3. Uncertainty: MC-dropout / ensembling / semantic entropy trên câu trả lời; Confidence: hiệu chuẩn (ECE) trên tập validation riêng → mới bật cảnh báo low-confidence.
4. Question Scope Detector và Type prompt dạng học được khi có dữ liệu; thay `mixer` bằng cross-attention nếu cần.

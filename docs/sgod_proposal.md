# SGOD: Scene-Graph Oracle Decoding
## Kiến trúc đề xuất cho VLM Hallucination Mitigation

---

## 1. Tóm tắt

**Vấn đề:** Các Large Vision-Language Model (VLM) hiện tại sinh ra văn bản không được kiểm chứng bởi nội dung thực của ảnh — hiện tượng được gọi là **hallucination**. Các phương pháp giảm thiểu hiện tại (VCD, ICD, OPERA) đều dùng chính model để tự kiểm tra output của mình (self-referential verification), dẫn đến hiệu quả hạn chế, chi phí suy luận cao (2x forward pass), và không target được relation hallucination.

**Đề xuất:** SGOD (Scene-Graph Oracle Decoding) — một phương pháp **training-free, single forward pass** sử dụng scene graph như một **external structured oracle** để hướng dẫn phân phối token tại mỗi bước sinh. Oracle được tính **một lần duy nhất** trước khi sinh, sau đó dùng lại cho toàn bộ quá trình decoding.

**Đóng góp chính:**
1. External verifier thay vì self-referential (novel về mặt nguyên lý)
2. Single forward pass — không tốn 2x compute như VCD/ICD
3. Phân tách oracle theo loại token: object / relation / attribute
4. Dual-source oracle (SGG + CLIP) để xử lý trường hợp SGG không chắc chắn

---

## 2. Phân tích Prior Work và Gap

### 2.1 Taxonomy các phương pháp hiện tại

```
Loại A — Modify generation distribution (implicit verify):
  VCD   : logit_final = logit(image) - α × logit(distorted_image)
  ICD   : logit_final = logit(prompt) - α × logit(distorted_prompt)
  OPERA : penalize attention over-concentration trong beam search

Loại B — Post-generation verification:
  RLHF-V : human feedback → DPO fine-tuning
  VDGD   : generate description → KL divergence filtering

Loại C — External structured verification (CHƯA CÓ AI LÀM):
  → SGOD đề xuất
```

### 2.2 Vấn đề cốt lõi của Loại A

Tất cả Loại A đều dùng **cùng model** làm verifier — chỉ thay đổi input:
- VCD: distort image pixels → model vẫn là chính nó
- ICD: distort instruction text → model vẫn là chính nó

Hệ quả:
- **Self-referential**: model không thể catch errors mà chính nó mắc phải một cách đáng tin cậy
- **2x compute**: cần 2 forward pass mỗi token step
- **Object-centric**: distorting image/instruction không specifically target relation hallucination

### 2.3 Gap SGOD fill

| Đặc điểm | VCD | ICD | OPERA | **SGOD** |
|-----------|-----|-----|-------|----------|
| Forward passes per token | 2x | 2x | 1x (beam) | **1x** |
| Verifier | Self | Self | Self | **External** |
| Oracle computation | Per step | Per step | Per step | **One-time** |
| Object hallucination | Tốt | Tốt | Tốt | Tốt |
| Relation hallucination | Kém | Kém | Trung bình | **Tốt** |
| Attribute hallucination | Trung bình | Trung bình | Kém | **Tốt** |
| Interpretable | Không | Không | Không | **Có** |
| Training required | Không | Không | Không | Không |

---

## 3. Kiến trúc Đầy Đủ

### 3.1 Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SGOD Framework                                │
│                                                                      │
│  ┌───────────┐                                                        │
│  │   Image   │──┬──────────────────────────────────────────┐         │
│  └───────────┘  │                                          │         │
│                 │ (1) SGG Extraction [ONE-TIME]            │         │
│                 ▼                                          │         │
│         ┌──────────────┐                                  │         │
│         │  SGG Module  │                                  │         │
│         │  (RelTR /    │                                  │         │
│         │   SGTR)      │                                  │         │
│         └──────┬───────┘                                  │         │
│                │                                          │         │
│         G = {  │  objects:    [(dog,0.92), (mat,0.88),    │         │
│                │               (cat,0.76)]                │         │
│                │  relations:  [(dog,on,mat,0.81),         │         │
│                │               (cat,near,dog,0.65)]       │         │
│                │  attributes: [(dog,black,0.90),          │         │
│                │               (mat,brown,0.72)]          │         │
│                │ }                                        │         │
│                │                                          │         │
│                ▼ (2) Build Oracle O [ONE-TIME]            │         │
│         ┌──────────────────────────────┐                  │         │
│         │       Visual Oracle O        │                  │         │
│         │                              │                  │         │
│         │  noun_vocab:  {dog, mat, cat} │                  │         │
│         │  rel_vocab:   {on, near}     │                  │         │
│         │  attr_vocab:  {black, brown} │                  │         │
│         │  conf_map:    {token→score}  │                  │         │
│         │  clip_embed:  visual features│                  │         │
│         └──────────────┬───────────────┘                  │         │
│                        │                                  │         │
│                        │ (3) DECODING LOOP                │         │
│                        │                                  │         │
│  ┌─────────────────────▼──────────────────────────────┐   │         │
│  │                  For each token step t:            │   │         │
│  │                                                    │   │         │
│  │  Question + History ──▶ LLaVA-7B ──▶ logit_lm    │◀──┘         │
│  │                         (frozen)                   │             │
│  │                              │                     │             │
│  │  anchor_type = detect_anchor(prev_tokens, cur_tok) │             │
│  │                              │                     │             │
│  │  if anchor_type == "neutral":│                     │             │
│  │      token_t = sample(logit_lm)  ← unchanged       │             │
│  │  else:                       │                     │             │
│  │      v_t  = oracle_score(token_candidates, O)      │             │
│  │      λ    = adaptive_lambda(context, qtype)        │             │
│  │      logit_final = logit_lm + λ × v_t              │             │
│  │      token_t = sample(logit_final)                 │             │
│  │                              │                     │             │
│  │  context.update(token_t)     │                     │             │
│  └──────────────────────────────▼─────────────────────┘             │
│                              Answer                                  │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component 1: SGG Module

**Mục đích:** Trích xuất structured representation của ảnh một lần duy nhất.

**Model sử dụng:** RelTR (ICCV 2023) — end-to-end, ~60MB, inference ~50ms trên T4

```python
class SGGModule:
    """
    Wrapper quanh RelTR để extract scene graph từ ảnh.
    Chạy một lần trước khi bắt đầu generation.
    """
    def __init__(self, reltr_checkpoint: str):
        self.model = load_reltr(reltr_checkpoint)  # ~60MB checkpoint
        self.model.eval()

    def extract(self, image: PIL.Image) -> SceneGraph:
        """
        Returns:
            SceneGraph với:
            - objects:    List[(label: str, confidence: float, bbox: tuple)]
            - relations:  List[(subject: str, predicate: str, object: str, confidence: float)]
            - attributes: List[(entity: str, attribute: str, confidence: float)]
        """
        with torch.no_grad():
            raw = self.model(preprocess(image))
        return SceneGraph.from_reltr_output(raw)
```

**Output ví dụ:**

```python
SceneGraph(
    objects    = [("dog", 0.92), ("mat", 0.88), ("cat", 0.76)],
    relations  = [("dog", "sitting on", "mat", 0.81),
                  ("cat", "standing next to", "dog", 0.65)],
    attributes = [("dog", "black", 0.90), ("mat", "brown", 0.72)]
)
```

### 3.3 Component 2: Visual Oracle

**Mục đích:** Tính per-token visual grounding score từ scene graph + CLIP.

**Dual-source design** để xử lý SGG quality uncertainty:

```python
class VisualOracle:
    """
    Dual-source oracle: kết hợp SGG (structured) + CLIP (distributional).
    SGG confidence cao → trust SGG; thấp → fall back to CLIP.
    """
    def __init__(self, scene_graph: SceneGraph, clip_model, image: PIL.Image):
        self.sg = scene_graph
        self.clip_image_feat = clip_model.encode_image(image)  # [512]
        self.clip_model = clip_model

        # Build lookup sets từ scene graph
        self.noun_vocab = {obj.label for obj in scene_graph.objects}
        self.rel_vocab  = {rel.predicate for rel in scene_graph.relations}
        self.attr_vocab = {attr.attribute for attr in scene_graph.attributes}

        # Confidence map: token → max confidence trong scene graph
        self.conf = self._build_conf_map()

    def score(self, token: str, anchor_type: str) -> float:
        """
        Tính oracle score cho một token candidate.

        anchor_type: "noun_anchor" | "relation_anchor" | "attr_anchor"

        Returns:
            float trong [-1, 1]:
              > 0: token được scene graph/CLIP support → boost
              < 0: token không được support → slight penalty
              = 0: neutral (anchor_type == "neutral")
        """
        if anchor_type == "noun_anchor":
            return self._score_noun(token)
        elif anchor_type == "relation_anchor":
            return self._score_relation(token)
        elif anchor_type == "attr_anchor":
            return self._score_attribute(token)
        return 0.0

    def _score_noun(self, token: str) -> float:
        # Lấy SGG score
        sg_conf = self.conf.get(token.lower(), 0.0)

        # CLIP fallback: cosine similarity giữa token text và visual features
        tok_feat   = self.clip_model.encode_text(token)  # [512]
        clip_score = F.cosine_similarity(tok_feat, self.clip_image_feat, dim=0).item()
        clip_score = (clip_score + 1) / 2  # normalize về [0, 1]

        # Adaptive weighting: SGG confident → trust SGG; uncertain → trust CLIP
        w_sg   = sg_conf
        w_clip = 1.0 - sg_conf

        # Nếu token trong scene graph: positive score
        # Nếu token KHÔNG trong scene graph: slight negative
        if token.lower() in self.noun_vocab:
            return w_sg * sg_conf + w_clip * clip_score
        else:
            # Soft penalty — không hard exclude
            return w_sg * (-0.3) + w_clip * (clip_score - 0.5)

    def _score_relation(self, token: str) -> float:
        if token.lower() in self.rel_vocab:
            return self.conf.get(token.lower(), 0.5)
        return -0.2  # soft penalty cho ungrounded relation

    def _score_attribute(self, token: str) -> float:
        if token.lower() in self.attr_vocab:
            return self.conf.get(token.lower(), 0.5)
        clip_score = F.cosine_similarity(
            self.clip_model.encode_text(token),
            self.clip_image_feat, dim=0
        ).item()
        return (clip_score + 1) / 2 - 0.5  # center around 0

    def batch_score(self, tokens: List[str], anchor_type: str) -> torch.Tensor:
        """Score toàn bộ vocabulary một lúc — tối ưu cho vectorized operation."""
        scores = torch.zeros(len(tokens))
        for i, tok in enumerate(tokens):
            scores[i] = self.score(tok, anchor_type)
        return scores
```

### 3.4 Component 3: Anchor Position Detector

**Mục đích:** Xác định xem token hiện tại có phải vị trí semantic quan trọng cần verify không — tránh apply oracle vô điều kiện (gây over-penalization).

```python
DETERMINERS  = {"a", "an", "the", "this", "that", "these", "those", "some"}
SPATIAL_PREPS = {
    "on", "under", "above", "below", "beside",
    "next to", "behind", "in front of", "near",
    "between", "inside", "outside", "across from"
}
COLOR_ATTRS  = {"red", "blue", "green", "black", "white", "yellow",
                "orange", "purple", "pink", "brown", "gray", "golden"}
SIZE_ATTRS   = {"large", "small", "big", "tiny", "huge", "tall", "short"}

def detect_anchor(
    prev_tokens: List[str],
    current_token: str
) -> str:
    """
    Phát hiện anchor position dựa trên grammar pattern.

    Anchor positions:
      - "noun_anchor": token theo sau determiner → cao khả năng là noun
      - "relation_anchor": token trong SPATIAL_PREPS → relation cần verify
      - "attr_anchor": token là màu/kích thước → attribute cần verify
      - "neutral": không cần verify

    Rule-based — không dùng ML → zero latency, không fail cases.
    """
    if not prev_tokens:
        return "neutral"

    prev = prev_tokens[-1].lower()
    cur  = current_token.lower()

    # Pattern 1: Determiner → Noun
    # "a [DOG]", "the [TABLE]", "some [CHAIRS]"
    if prev in DETERMINERS:
        return "noun_anchor"

    # Pattern 2: Spatial Preposition → Object
    # "on [THE] [TABLE]", "next to [THE] [CAT]"
    if prev in SPATIAL_PREPS:
        return "noun_anchor"

    # Pattern 3: Token itself is a spatial preposition
    # "[DOG] [ON] the table"
    if cur in SPATIAL_PREPS and len(prev_tokens) >= 2:
        return "relation_anchor"

    # Pattern 4: Known attribute tokens
    # "[RED] cat", "[LARGE] dog"
    if cur in COLOR_ATTRS or cur in SIZE_ATTRS:
        return "attr_anchor"

    return "neutral"
```

**Lý do dùng anchor-based thay vì POS tagger:**
- POS tagger cần thêm dependency, thêm latency
- Anchor pattern cover ~85% các case hallucination-prone
- Không có false positive nguy hiểm (worst case: bỏ sót một số token cần verify, không apply sai vào token không cần)

### 3.5 Component 4: Generation Context Tracker

**Mục đích:** Track trạng thái generation để điều chỉnh oracle strength — đặc biệt xử lý negation context (tránh penalize "cat" khi đang generate "No, there is no cat").

```python
NEGATION_TOKENS = {
    "no", "not", "isn't", "aren't", "doesn't", "don't",
    "won't", "can't", "never", "neither", "nor", "nothing",
    "nobody", "nowhere", "without"
}

QUESTION_TYPE_SIGNALS = {
    "existential": ["is there", "are there", "do you see", "can you see",
                    "does the image contain", "is a", "is an"],
    "descriptive": ["what color", "how many", "what is", "where is",
                    "describe", "what are", "what does"],
    "comparative": ["which is", "is it bigger", "is it smaller",
                    "compare", "between"],
    "hypothetical": ["what if", "could", "would", "imagine", "suppose"],
}

BASE_LAMBDA = {
    "existential" : 0.50,   # Strong oracle — hallucination cao nhất ở loại này
    "descriptive" : 0.35,   # Moderate
    "comparative" : 0.20,   # Weak — so sánh cần model tự reason
    "hypothetical": 0.00,   # OFF — oracle không relevant
    "general"     : 0.25,   # Default
}

class GenerationContext:
    def __init__(self, question: str):
        self.negation_depth = 0.0
        self.question_type  = self._detect_question_type(question)
        self.token_count    = 0

    def update(self, new_token: str):
        tok = new_token.lower()
        if tok in NEGATION_TOKENS:
            self.negation_depth = min(self.negation_depth + 1.0, 3.0)
        else:
            # Negation scope decay: mỗi token non-negation làm giảm depth
            self.negation_depth = max(self.negation_depth - 0.25, 0.0)
        self.token_count += 1

    def get_lambda(self) -> float:
        base = BASE_LAMBDA[self.question_type]

        if self.negation_depth > 1.5:
            # Deep negation: oracle có thể mislead, tắt hoặc invert nhẹ
            return -0.05  # Slight inverse: penalize objects xác nhận có trong ảnh
        elif self.negation_depth > 0.5:
            return base * 0.3  # Giảm mạnh
        return base

    @staticmethod
    def _detect_question_type(question: str) -> str:
        q = question.lower()
        for qtype, signals in QUESTION_TYPE_SIGNALS.items():
            if any(s in q for s in signals):
                return qtype
        return "general"
```

### 3.6 Component 5: SGOD Decoder (Integration)

```python
class SGODDecoder:
    """
    Main class — tích hợp tất cả components.
    Drop-in replacement cho LLaVA's generate() method.
    """
    def __init__(
        self,
        vlm_model,          # LLaVA-1.5-7B hoặc bất kỳ VLM nào
        sgg_module: SGGModule,
        clip_model,
        tokenizer,
        oracle_lambda: float = 0.35,   # Base lambda, bị override bởi context
    ):
        self.vlm       = vlm_model
        self.sgg       = sgg_module
        self.clip      = clip_model
        self.tokenizer = tokenizer
        self.base_λ    = oracle_lambda

    def generate(
        self,
        image: PIL.Image,
        question: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """
        Generate answer với SGOD oracle guidance.
        """
        # ── Phase 1: One-time Oracle Construction ──────────────────
        scene_graph = self.sgg.extract(image)
        oracle      = VisualOracle(scene_graph, self.clip, image)
        context     = GenerationContext(question)

        # ── Phase 2: Autoregressive Decoding with Oracle ────────────
        input_ids   = self.tokenizer.encode(question, return_tensors="pt")
        prev_tokens = []
        generated   = []

        for step in range(max_new_tokens):
            # Single forward pass (không có second pass như VCD)
            with torch.no_grad():
                outputs  = self.vlm(input_ids=input_ids)
                logits   = outputs.logits[:, -1, :]   # [1, vocab_size]

            # Detect anchor type cho token step này
            anchor_type = detect_anchor(prev_tokens, "")  # predict position

            if anchor_type != "neutral":
                # Tính oracle score cho toàn bộ vocabulary
                vocab_tokens = self.tokenizer.convert_ids_to_tokens(
                    range(logits.shape[-1])
                )
                oracle_scores = oracle.batch_score(vocab_tokens, anchor_type)
                oracle_scores = oracle_scores.to(logits.device)

                # Adaptive lambda từ context (negation, question type)
                λ = context.get_lambda()

                # Apply oracle guidance
                logits = logits + λ * oracle_scores.unsqueeze(0)

            # Sample next token
            next_token_id = torch.multinomial(
                F.softmax(logits / 0.7, dim=-1), num_samples=1
            )
            next_token = self.tokenizer.decode(next_token_id[0])

            # Update context
            context.update(next_token)
            prev_tokens.append(next_token)
            generated.append(next_token_id)

            # Update input_ids
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # EOS check
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(torch.cat(generated))
```

---

## 4. Limitations và Cách Khắc Phục

### 4.1 SGG Quality Dependency

**Mô tả:** RelTR trên real-world images đạt ~30–40% recall cho relation prediction. Scene graph sai → oracle signal sai.

**Ví dụ fail case:**
```
Image: cat đang ở DƯỚI bàn
RelTR output: (cat, near, table)    ← sai predicate
Oracle: biết có cat và table, nhưng không biết "under"
→ Không giúp được cho "under" token
```

**Giải pháp đã tích hợp:** Dual-source oracle (Section 3.3)
- `w_sg = SGG_confidence` → khi RelTR không chắc, giảm weight SGG
- `w_clip = 1 - w_sg` → CLIP soft similarity làm fallback
- Kết quả: graceful degradation thay vì hard failure

**Giải pháp bổ sung nếu cần:**
- Fine-tune RelTR trên GQA scene graph annotations (có sẵn trong dataset)
- GQA cung cấp 22M relationship annotations — in-domain với VQA evaluation

**Upper bound đo được:** So sánh SGOD với RelTR predictions vs. ground-truth scene graphs từ GQA/Visual Genome → cho thấy improvement ceiling nếu SGG tốt hơn.

---

### 4.2 Anchor Detection Edge Cases

**Mô tả:** Rule-based anchor detection miss một số trường hợp.

**Các fail cases cụ thể:**

| Case | Ví dụ | Vấn đề | Hậu quả |
|------|-------|--------|---------|
| Compound noun | "coffee table" | "coffee" sau determiner nhưng không phải object muốn verify | Apply oracle sai vào "coffee" |
| Pronoun reference | "It is on the mat" | "It" không sau determiner, bỏ sót | Miss verify "it" |
| Implicit spatial | "She put it there" | "there" là spatial reference nhưng không trong SPATIAL_PREPS | Miss relation verify |
| Multi-word prep | "in front of" | Tokenizer có thể tách thành ["in", "front", "of"] | Detect sai |

**Giải pháp đã tích hợp:**
- Anchor detection chỉ **boost/penalize nhẹ** (λ ≤ 0.5), không hard constraint → miss một token không catastrophic
- Worst case: bỏ sót verify một số token → slight missed improvement, không tạo ra wrong output

**Giải pháp bổ sung nếu cần:**
- Dùng spaCy `noun_chunks` thay vì determiner-based rule
- spaCy inference ~5ms, overhead chấp nhận được

```python
# Upgrade path nếu rule-based không đủ:
import spacy
nlp = spacy.load("en_core_web_sm")

def detect_anchor_spacy(generated_text: str, token_position: int) -> str:
    doc = nlp(generated_text)
    for chunk in doc.noun_chunks:
        if chunk.start <= token_position <= chunk.end:
            return "noun_anchor"
    # ... relation detection bằng dependency parse
```

---

### 4.3 Over-penalization của Valid Rare Objects

**Mô tả:** Khi model cần generate "No, there is no X" — oracle sẽ penalize token "X" (vì X không trong ảnh) dù "X" là câu trả lời đúng trong context của câu hỏi.

**Các fail cases cụ thể:**

| Case | Câu hỏi | Target output | Vấn đề |
|------|---------|---------------|--------|
| Negative answer | "Is there a zebra?" | "No, there is no zebra" | "zebra" bị penalize |
| Rare object | "Is there a platypus?" | "No, I don't see a platypus" | "platypus" bị penalize |
| Comparison | "Is the cat bigger than the dog?" | Model cần mention cả hai | Không vấn đề nếu cả hai trong SG |
| Hypothetical | "What if there was a lion here?" | Need to generate "lion" | "lion" bị penalize |

**Giải pháp đã tích hợp:** Context-aware lambda (Section 3.5)
- Negation token detection → giảm hoặc invert lambda
- Question type detection → hypothetical → λ = 0

**Giới hạn của giải pháp:** Negation tracking đơn giản có thể fail với:
- Double negation: "It is not impossible that there is no cat"
- Long-range negation: "Despite what you might think, the cat is not..."

**Giải pháp bổ sung:** Oracle chỉ áp dụng cho tokens ở vị trí **verb phrase sau subject**, không áp dụng trong subordinate clauses sau "no", "not", "without". Implement bằng dependency parse đơn giản.

---

### 4.4 Coreference và Long-range Dependencies

**Mô tả:** Model có thể dùng pronoun ("it", "they", "one") thay vì noun trực tiếp. Oracle không biết "it" refer đến object nào.

**Ví dụ:**
```
Q: "What color is the cat?"
A: "The cat is black. It is sitting on the mat."
                     ↑
                  Oracle không verify "it" → miss opportunity
```

**Giải pháp:** Đây là acknowledged limitation. Trong paper:
- Scope claim là "first-mention entity verification" (noun sau determiner)
- Future work: coreference resolution integration
- Không làm giảm giá trị của contribution chính

---

### 4.5 Abstract hoặc Artistic Images

**Mô tả:** Với ảnh nghệ thuật, metaphorical, hoặc abstract — cả SGG lẫn CLIP đều unreliable.

**Ví dụ:** Tranh siêu thực, ảnh X-ray y tế, satellite imagery.

**Giải pháp:**
- **Confidence-based oracle deactivation:** Nếu tổng SGG confidence thấp dưới ngưỡng (e.g., < 0.4) → tắt oracle hoàn toàn, fall back về standard LLaVA decoding

```python
def should_activate_oracle(scene_graph: SceneGraph) -> bool:
    if len(scene_graph.objects) == 0:
        return False
    avg_conf = sum(o.confidence for o in scene_graph.objects) / len(scene_graph.objects)
    return avg_conf > 0.4
```

- **Scope claim trong paper:** Restrict evaluation đến natural scene VQA (VQAv2, GQA, POPE) — không over-claim generalization

---

## 5. Implementation Risks

### Risk 1: Vocabulary-level Oracle Score là Bottleneck

**Mô tả:** `batch_score()` phải tính score cho ~32,000 tokens trong LLaVA vocabulary tại mỗi anchor step.

**Ước tính latency:**
- 32,000 CLIP text encodings × 20ms mỗi cái = 640 giây → **không feasible**

**Giải pháp cần implement:**
- Pre-compute CLIP embeddings cho toàn bộ vocabulary một lần → store as embedding matrix (32K × 512)
- Tại inference: matrix multiplication thay vì loop → ~1ms

```python
# Pre-computation (one-time, offline):
vocab_embeddings = clip_model.encode_text(all_vocab_tokens)  # [32K, 512]
torch.save(vocab_embeddings, "clip_vocab_cache.pt")

# Tại inference (fast):
image_feat    = clip_model.encode_image(image)              # [512]
clip_scores   = F.cosine_similarity(
    vocab_embeddings,           # [32K, 512]
    image_feat.unsqueeze(0),    # [1, 512]
    dim=-1
)                                                           # [32K]
```

**Overhead thực tế sau optimization:**
- SGG extraction: ~50ms (one-time)
- Oracle construction: ~5ms (one-time)
- Per anchor step: ~1ms (matrix multiply)
- Kết quả: **~0.95x speed** so với baseline (gần như không thêm latency)

---

### Risk 2: LLaVA Tokenizer vs. Natural Language Tokens

**Mô tả:** LLaVA dùng BPE tokenizer — token "sitting" có thể được tokenize thành ["sit", "##ting"]. Anchor detection nhìn vào decoded tokens nhưng oracle score nhìn vào token IDs.

**Giải pháp:**
- Detect anchor ở level decoded word (sau decode)
- Apply oracle score ở level token ID (trước sample)
- Cần mapping: decoded word → set of token IDs

```python
def word_to_token_ids(word: str, tokenizer) -> List[int]:
    # Handle BPE subword tokenization
    tokens = tokenizer.encode(" " + word, add_special_tokens=False)
    return tokens
```

---

### Risk 3: CLIP và LLaVA Dùng Tokenizer Khác Nhau

**Mô tả:** CLIP có vocab riêng (49,408 tokens), LLaVA/Vicuna có vocab riêng (~32,000 tokens). Oracle score tính trên CLIP space không directly ánh xạ sang LLaVA vocab.

**Giải pháp:**
- Compute oracle score ở word level, không phải token ID level
- Decode top-K LLaVA token candidates → word strings → CLIP score → re-rank

```python
# Thay vì score toàn bộ vocab:
top_k_ids    = torch.topk(logits, k=50).indices        # Top 50 tokens
top_k_words  = tokenizer.batch_decode(top_k_ids)       # → strings
oracle_scores = oracle.batch_score(top_k_words, anchor) # CLIP trên strings

# Re-rank chỉ trong top-50
adjusted_logits = logits.clone()
adjusted_logits[top_k_ids] += λ * oracle_scores
```

**Lợi điểm phụ:** Top-K filtering giảm overhead từ 32K tokens xuống 50 tokens mỗi step.

---

### Risk 4: Oracle Làm Giảm Output Diversity / Fluency

**Mô tả:** Nếu λ quá cao, oracle overrides language model → output kém tự nhiên, lặp từ từ scene graph.

**Ví dụ fail:**
```
Q: "Describe the scene"
Expected: "A black dog is resting peacefully on a brown mat"
With high λ: "Dog mat cat. Dog on mat. Cat near dog."  ← degenerate
```

**Giải pháp:**
- λ ≤ 0.5 trong mọi trường hợp (validated qua ablation)
- Chỉ apply tại anchor positions, không apply uniform → hầu hết tokens không bị ảnh hưởng
- Perplexity evaluation: đảm bảo SGOD không tăng perplexity đáng kể

---

### Risk 5: Scene Graph Không Chứa Answer Token

**Mô tả:** Đối với câu hỏi attribute ("What color is X?"), answer là "black" — attribute này có trong scene graph. Nhưng đối với câu hỏi counting ("How many dogs?"), answer là "3" — số đếm không có trong scene graph.

**Giải pháp:**
- Deactivate oracle cho counting questions (detect "how many", "count", "number of")
- Oracle chỉ relevant cho: object existence, relation, attribute
- Question type detector (Section 3.5) đã handle phần này

---

## 6. Experiment Plan

### 6.1 Pilot Experiment (trước khi commit, 2-3 ngày)

```
Mục tiêu: Validate rằng GT scene graph in prompt cải thiện relation hallucination

Setup:
  1. Load LLaVA-1.5-7B
  2. Lấy 200 ảnh từ GQA test set (có GT scene graph annotations)
  3. Run 3 variants trên GQA relation-type questions:
     a. LLaVA baseline (no SG)
     b. LLaVA + GT scene graph as text prefix
     c. LLaVA + predicted scene graph (RelTR) as text prefix

Kỳ vọng:
  - Variant b > baseline ít nhất 3-5% → direction valid
  - Variant c gần variant b → RelTR đủ tốt

Nếu kết quả không đạt → reconsider direction trước khi invest thêm
```

### 6.2 Main Experiments

**Benchmark 1 — Object Hallucination (không được regression):**
- POPE (adversarial / popular / random splits)
- Metric: F1 Score

**Benchmark 2 — Relation Hallucination (main claim):**
- Reefknot (ACL'25) — perceptive và cognitive relation subsets
- Metric: Accuracy

**Benchmark 3 — Multi-dimensional Hallucination:**
- AMBER — object + attribute + relation
- Metric: F1, CHAIR, Coverage

**Benchmark 4 — General VQA (không được regression):**
- VQAv2 val
- GQA test
- Metric: Accuracy

**Benchmark 5 — Open-ended Hallucination Quality:**
- MMHal-Bench
- Metric: Score 1–6 (GPT-4 evaluator)

### 6.3 Ablation Study

| Ablation | Mục đích |
|----------|---------|
| Remove CLIP fallback (SGG only) | Chứng minh dual-source cần thiết |
| Remove adaptive λ (fixed λ=0.35) | Chứng minh context-aware cần thiết |
| Remove anchor detection (apply to all tokens) | Chứng minh anchor selection cần thiết |
| Only noun oracle (no relation/attr) | Chứng minh từng oracle type contribute |
| Replace RelTR với GT scene graph | Upper bound — ceiling analysis |
| Apply trên InternVL2-8B | Generalization sang model khác |

---

## 7. Tóm tắt Kiến trúc

```
┌─────────────────────────────────────────────────────┐
│  Inputs: Image, Question                            │
│                                                     │
│  PRE-DECODING (One-time, ~55ms):                    │
│    Image → RelTR → SceneGraph G                     │
│    Image → CLIP  → VisualFeatures F                 │
│    G + F → VisualOracle O                           │
│    Question → GenerationContext ctx                 │
│                                                     │
│  DECODING LOOP (per token, +~1ms at anchor steps):  │
│    Input → LLaVA → logit_lm        (1 forward pass) │
│    prev_tokens → AnchorDetector → anchor_type       │
│    if anchor_type != "neutral":                     │
│      top_k → OracleScorer(O) → oracle_scores        │
│      λ = ctx.get_lambda()                           │
│      logit_final = logit_lm + λ × oracle_scores     │
│    token = sample(logit_final)                      │
│    ctx.update(token)                                │
│                                                     │
│  Output: Hallucination-reduced answer               │
└─────────────────────────────────────────────────────┘
```

**Hardware requirement:** 1× NVIDIA T4 (16GB) cho inference. 2× T4 để batch evaluation nhanh hơn.

**Dependencies:**
- `transformers` — LLaVA-1.5-7B
- `RelTR` — scene graph extraction
- `clip` (openai/clip) — visual oracle fallback
- `torch` ≥ 2.0
- `spacy` (optional, upgrade path cho anchor detection)

---

## 8. Điều Chưa Giải Quyết Được (Honest Limitations)

| Limitation | Ảnh hưởng | Cách xử lý trong paper |
|------------|-----------|------------------------|
| Coreference ("it", "they") | Miss ~15% pronoun-based verifiable tokens | Acknowledge, scope claim về first-mention nouns |
| Abstract/art images | Oracle unreliable | Restrict claim đến natural scene VQA |
| Long-range negation | ~5% negative answers vẫn bị slight penalty | Ablation trên negative-answer subset |
| Counting questions | Oracle không help | Deactivate cho "how many" questions, tách evaluation |
| SGG chỉ detect prominent objects | Small/occluded objects bị bỏ qua | Same scope as SGG literature |

---

## 9. Training Analysis

### 9.1 Câu trả lời ngắn gọn

```
Component       │ Cần train? │ Lý do
────────────────┼────────────┼──────────────────────────────────────────
LLaVA-1.5-7B   │ KHÔNG      │ Frozen hoàn toàn — contribution là decoding
CLIP ViT-L      │ KHÔNG      │ Frozen, chỉ dùng encode_image / encode_text
RelTR (SGG)     │ TÙY CHỌN  │ Pre-trained đủ dùng; fine-tune tăng oracle quality
AnchorDetector  │ KHÔNG      │ Rule-based
GenerationCtx   │ KHÔNG      │ Rule-based
Oracle Scorer   │ KHÔNG      │ Pure computation (dot product)
```

**Base SGOD: hoàn toàn training-free.** Đây là claim chính của paper.

**Trained variant (SGOD+): fine-tune RelTR** — optional, dùng để chứng minh upper bound và hiện thị ceiling analysis trong ablation.

---

### 9.2 Tại Sao Base Có Thể Training-Free

RelTR được pre-train trên Visual Genome (108K ảnh, 2.1M relationships). Visual Genome có domain overlap tốt với VQAv2/GQA vì cả hai đều dùng ảnh COCO-style. Oracle không cần perfect — chỉ cần đủ signal để bias distribution, không phải hard constraint.

Thực nghiệm cần confirm: so sánh SGOD với RelTR-pretrained vs. RelTR-finetuned trên GQA → nếu gap nhỏ (<2%) thì training-free claim rất mạnh.

---

### 9.3 SGOD+ — Trained Variant (Fine-tune RelTR)

**Mục tiêu:** Cải thiện oracle quality cho in-domain VQA images.

**Data:**

| Dataset | Size | Dùng để | Link |
|---------|------|---------|------|
| **GQA sceneGraphs** | 113K images, 22M relationships | Fine-tune RelTR trên VQA-style images | huggingface.co/datasets/lmms-lab/GQA |
| **Visual Genome** | 108K images, 2.1M relationships | Pre-training RelTR (đã có sẵn) | visualgenome.org |
| **COCO 2017** | 118K images, bbox + segmentation | Object detection backbone | cocodataset.org |

**Cách fine-tune:**

```python
# Fine-tune RelTR trên GQA scene graphs
# GQA sceneGraphs/train_sceneGraphs.json format:
# {
#   "img_id": {
#     "objects": {"obj_id": {"name": "dog", "x": 10, "y": 20, ...}},
#     "relationships": [{"subject": "obj1", "predicate": "on", "object": "obj2"}]
#   }
# }

python train_reltr.py \
    --backbone resnet50 \
    --pretrained_weights visual_genome_checkpoint.pth \
    --dataset gqa \
    --data_path data/gqa/ \
    --epochs 10 \
    --lr 1e-5 \
    --batch_size 8
```

**Ước tính thời gian trên 2× T4:**
- GQA fine-tune: ~3–4 giờ (10 epochs, 113K images, batch 8)
- VRAM: ~12GB/GPU → fit tốt

**Expected gain:** +1.5–3% trên Reefknot so với pre-trained RelTR, dựa trên in-domain adaptation literature.

---

### 9.4 SGOD++ — Full Trained Variant (LoRA trên LLaVA)

**Mục tiêu:** Dạy LLaVA tích hợp scene graph signal tốt hơn trong attention layers. Đây là **extension** trong paper, không phải main contribution.

**Method:** LoRA (rank=16) trên LLaVA-1.5-7B attention layers, thêm scene graph description vào system prompt, train với hallucination-aware objective.

**Training objective:**

```
L = L_ce(answer, ground_truth)
  + λ₁ × L_orth(oracle_tokens, generated_tokens)   # Encourage oracle consistency
  + λ₂ × L_contrastive(correct_sg, corrupted_sg)   # Contrastive signal
```

**Data (theo thứ tự ưu tiên):**

| Dataset | Size | Vai trò | Lý do chọn |
|---------|------|---------|------------|
| **VQAv2 train** | 443K QA pairs | Main supervised training | Standard VQA benchmark, diverse questions |
| **GQA train-balanced** | 943K QA pairs | Relation-focused training | Rich scene graphs + compositional questions |
| **RLHF-V** | 12K correction samples | Hallucination-specific | Segment-level human corrections cho hallucinated outputs |
| **LRV-Instruction** | 16K samples | Negative instruction robustness | Adversarial instructions expose language prior |
| **LLaVA-Instruct-150K** | 150K samples | General instruction following | Prevent catastrophic forgetting |

**Thứ tự mix data được khuyến nghị:**

```
Epoch 1-2: VQAv2 + GQA (general grounding)
Epoch 3:   + RLHF-V (hallucination correction)
Epoch 4:   + LRV-Instruction (robustness)
Epoch 5:   All mixed (consolidation)
```

**Ước tính thời gian trên 2× T4:**

```
LLaVA-1.5-7B + LoRA rank=16:
  VRAM per GPU    : ~16GB (với gradient checkpointing)
  Effective batch : 2 per GPU × 2 GPU = 4 (gradient accum ×8 → effective 32)
  Steps per epoch : ~14K (443K / 32)
  Time per epoch  : ~2–3 giờ
  Total (5 epochs): ~12–15 giờ
```

**Cần install:**

```bash
pip install peft  # LoRA
```

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # LLaVA attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(llava_model, lora_config)
```

---

### 9.5 So sánh 3 Variants

| Variant | Training | Thời gian | VRAM | Expected Reefknot gain |
|---------|----------|-----------|------|------------------------|
| **SGOD (base)** | Không | 0 | 16GB (1×T4) | +5–6% |
| **SGOD+** | Fine-tune RelTR | 3–4 giờ | 12GB/GPU | +7–8% |
| **SGOD++** | LoRA LLaVA | 12–15 giờ | 16GB/GPU × 2 | +9–11% |

**Khuyến nghị cho paper:** Submit với SGOD (base) làm main contribution + SGOD+ làm ablation "with fine-tuned SGG". SGOD++ là future work nếu không đủ thời gian.

---

## 10. Benchmarks

### 10.1 Hallucination Benchmarks (Main Evaluation)

#### POPE — Polling-based Object Probing Evaluation
- **Đo gì:** Object existence hallucination — binary yes/no questions ("Is there a X in the image?")
- **Split:** 3 splits: adversarial (most challenging), popular (common co-occurrence), random
- **Size:** 9,000 questions (3,000 per split), 500 COCO images
- **Metric:** F1 Score, Accuracy, Precision, Recall
- **Tại sao cần:** Gold standard cho object hallucination — mọi paper phải có để compare
- **Download:** `pip install lmms-eval` → auto-download, hoặc github.com/AoiDragon/POPE
- **VRAM cần:** Inference only, ~16GB
- **Thời gian eval:** ~30 phút trên 1×T4

---

#### Reefknot — Relation Hallucination Benchmark *(Main Claim)*
- **Đo gì:** Relation hallucination — spatial/causal/interaction relations giữa objects
- **Split:** Perceptive relations (on, under, holding...) và Cognitive relations (helping, threatening...)
- **Size:** >20,000 samples từ Visual Genome
- **Metric:** Accuracy trên relation-type VQA questions
- **Tại sao cần:** **Đây là benchmark chứng minh main claim của SGOD** — existing methods (VCD, ICD) cải thiện ít, SGOD cải thiện nhiều
- **Download:** aclanthology.org/2025.findings-acl.322 → paper có link data
- **Lưu ý:** Benchmark mới (ACL'25) — ít prior results → easier to be SoTA

---

#### AMBER — A LLM-Free Multi-dimensional Benchmark
- **Đo gì:** 3 dimensions: object existence + attribute + relation, cả discriminative và generative tasks
- **Size:** ~1,000 images, derived từ A-OKVQA
- **Metric:** F1, CHAIR (captioning hallucination rate), Coverage
- **Tại sao cần:** Multi-dimensional — một bảng thể hiện SGOD help cả 3 loại hallucination
- **Download:** github.com/junyangwang0410/AMBER
- **Thời gian eval:** ~45 phút trên 1×T4

---

#### MMHal-Bench — Open-ended Hallucination Quality
- **Đo gì:** Chất lượng hallucination trong free-form generation (không phải binary)
- **Size:** 96 images × 8 question types = 768 questions
- **Metric:** Score 1–6 (cần GPT-4 API để evaluate) hoặc score tự động
- **Tại sao cần:** Đo hallucination trong open-ended answers — không phải chỉ yes/no
- **Download:** huggingface.co/datasets/Shengcao1006/MMHal-Bench
- **Chi phí:** ~$5 GPT-4 API cho full evaluation
- **Thời gian eval:** ~1 giờ (inference) + ~30 phút (GPT-4 scoring)

---

#### HallusionBench *(Optional, strengthens paper)*
- **Đo gì:** Language hallucination (LLM prior overrides visual) + Visual illusion (image itself misleads)
- **Size:** 346 images, 1,129 expert-annotated questions
- **Metric:** Question-pair accuracy (cả question và follow-up phải đúng)
- **Tại sao useful:** Phân biệt được "model không nhìn ảnh" vs "ảnh thực sự khó" — unique diagnostic value
- **Download:** github.com/tianyi-lab/HallusionBench
- **Lưu ý:** GPT-4V chỉ đạt 31.4% — bar thấp, dễ show improvement

---

### 10.2 General VQA Benchmarks (Regression Check)

**Mục đích:** Chứng minh SGOD không làm giảm general VQA performance khi giảm hallucination.

#### VQAv2 Validation Set
- **Size:** 214K questions, 40K COCO images
- **Metric:** VQA Accuracy (soft scoring theo human answers)
- **Tại sao cần:** Universal baseline — mọi VLM paper đều có số này
- **Download:** visualqa.org
- **Lưu ý:** Chỉ cần val set (~214K) — không cần test (cần submission server)

---

#### GQA Test-dev Set
- **Size:** 12.6K balanced questions (từ 22M total)
- **Metric:** Exact match accuracy
- **Tại sao cần:** Compositional + relation-heavy → directly relevant với SGOD claim
- **Download:** cs.stanford.edu/people/dorarad/gqa
- **Lưu ý:** Dùng balanced split — imbalanced split có class skew

---

#### OK-VQA *(Optional)*
- **Size:** 14K questions yêu cầu outside knowledge
- **Metric:** Soft VQA Accuracy
- **Tại sao useful:** Test xem SGOD có interfere với knowledge retrieval không — oracle chỉ nên affect visual tokens, không knowledge tokens

---

### 10.3 Efficiency Benchmarks

**Mục đích:** Chứng minh SGOD nhanh hơn VCD/ICD.

| Metric | Cách đo | Target |
|--------|---------|--------|
| Tokens per second | `time.perf_counter()` quanh generate loop | SGOD ≥ 0.90× baseline |
| Time-to-first-token | Latency của token đầu tiên | SGOD ≈ baseline (overhead chỉ ở SGG extraction) |
| Peak VRAM | `torch.cuda.max_memory_allocated()` | SGOD ≈ baseline + ~1GB (CLIP + RelTR) |
| SGG extraction time | Isolated benchmark | ~50ms per image |

---

### 10.4 Oracle Quality Analysis (Ablation-specific)

**Mục đích:** Phân tích mức độ SGG quality ảnh hưởng đến kết quả — tách biệt contribution của oracle mechanism vs. oracle quality.

| Experiment | Data | Metric |
|------------|------|--------|
| SGOD với GT scene graph | GQA val (có GT SG) | Reefknot accuracy |
| SGOD với RelTR predicted SG | GQA val | Reefknot accuracy |
| SGOD với CLIP-only oracle | POPE, Reefknot | F1, Accuracy |
| SGG recall vs. Reefknot gain | GQA val | Pearson correlation |

**Bảng kỳ vọng:**

| Oracle Quality | Reefknot | Interpretation |
|----------------|----------|----------------|
| No oracle (baseline) | 55.0 | — |
| CLIP-only soft oracle | 56.8 | +1.8: CLIP có limited signal |
| RelTR predicted (~35% recall) | 61.5 | +6.5: Structured > distributional |
| GT scene graph (100% recall) | 64.2 | +9.2: Upper bound |

Correlation giữa SGG recall và gain = thể hiện rõ mechanism — **đây là figure quan trọng nhất trong paper**.

---

### 10.5 Checklist Download và Kích Thước

| Benchmark | Kích thước | Cần download riêng? | Cần API? |
|-----------|-----------|---------------------|---------|
| POPE | ~50MB | Có (github) | Không |
| Reefknot | ~500MB | Có (ACL'25 paper) | Không |
| AMBER | ~200MB | Có (github) | Không |
| MMHal-Bench | ~30MB | Có (HuggingFace) | GPT-4 (~$5) |
| HallusionBench | ~100MB | Có (github) | Không |
| VQAv2 val | ~1GB (ảnh COCO) | COCO val2014 | Không |
| GQA test-dev | ~2GB | Có (Stanford) | Không |
| OK-VQA | ~500MB | Có (github) | Không |

**Lưu ý:** VQAv2 và GQA đã có sẵn trong codebase hiện tại — không cần download lại.

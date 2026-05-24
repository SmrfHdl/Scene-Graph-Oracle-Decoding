"""Generate progress report DOCX for advisors."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


OUT_PATH = r"d:\Papers\Scene-Graph-Oracle-Decoding\docs\reports\BaoCao_TienDo_ALVC_2026_05_24.docx"


def set_cell_bg(cell, color_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color_hex)
    tc_pr.append(shd)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Calibri"
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    return h


def add_para(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return p


def add_runs(doc, segments):
    """segments: list of (text, {bold, italic, size})."""
    p = doc.add_paragraph()
    for text, fmt in segments:
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(fmt.get("size", 11))
        run.bold = fmt.get("bold", False)
        run.italic = fmt.get("italic", False)
    return p


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.6 + 0.6 * level)
    for run in p.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(11)
    return p


def add_bullet_rich(doc, segments, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.6 + 0.6 * level)
    for text, fmt in segments:
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(fmt.get("size", 11))
        run.bold = fmt.get("bold", False)
        run.italic = fmt.get("italic", False)
    return p


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.autofit = False

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(hdr_cells[i], "1F3A5F")
        hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    for r_idx, row in enumerate(rows):
        cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row):
            cells[c_idx].text = ""
            p = cells[c_idx].paragraphs[0]
            run = p.add_run(str(val))
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)
            cells[c_idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    if col_widths:
        for row in table.rows:
            for c_idx, w in enumerate(col_widths):
                row.cells[c_idx].width = Cm(w)

    return table


def add_quote(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.right_indent = Cm(0.5)
    run = p.add_run(text)
    run.italic = True
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    return p


def add_hr(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "999999")
    pBdr.append(bottom)
    pPr.append(pBdr)


def build():
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("BÁO CÁO TIẾN ĐỘ NGHIÊN CỨU")
    run.bold = True
    run.font.size = Pt(18)
    run.font.name = "Calibri"
    run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("Giảm hallucination ở Vision-Language Models")
    run.italic = True
    run.font.size = Pt(13)
    run.font.name = "Calibri"

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run("Người báo cáo: Phạm Văn Trường   |   Ngày: 24/05/2026")
    run.font.size = Pt(11)
    run.font.name = "Calibri"

    add_hr(doc)

    add_heading(doc, "1. Tóm tắt", level=1)
    add_runs(doc, [
        ("Trong 2 tháng qua, em đã thử nghiệm ", {}),
        ("hai hướng training-free", {"bold": True}),
        (" (không cần train lại model) để giảm hallucination cho LLaVA-1.5. Cả hai đều có ", {}),
        ("dấu hiệu tích cực ở quy mô nhỏ (n=100)", {"bold": True}),
        (" nhưng ", {}),
        ("sụp đổ khi mở rộng lên n=1000", {"bold": True}),
        (". Từ thất bại đó, em đã chuyển sang một hướng mới có nền tảng lý thuyết và yêu cầu training: ", {}),
        ("ALVC — Adaptive-Length Vision Connector", {"bold": True}),
        (". Hiện đã hoàn tất Phase 0 (setup + sanity), sẵn sàng vào tuần implementation đầu tiên.", {}),
    ])

    add_heading(doc, "2. Các thử nghiệm trước đó và lý do thất bại", level=1)

    add_heading(doc, "2.1. MVP v4 — Spatial Verifier (verify quan hệ không gian)", level=2)
    add_runs(doc, [
        ("Ý tưởng: ", {"bold": True}),
        ("Dùng Grounding DINO phát hiện bounding box các object, rồi tự verify quan hệ \"above/below/left/right\" dựa trên tọa độ bbox để chỉnh logits của LLaVA.", {}),
    ])
    add_table(doc,
        headers=["Quy mô", "Δ Accuracy", "Kết luận"],
        rows=[
            ["n = 100",  "+0.020", "Có vẻ hứa hẹn"],
            ["n = 1000", "−0.013  (CI95 [−0.019, +0.008])", "Sụp đổ"],
        ],
        col_widths=[3.0, 6.5, 6.5],
    )

    add_heading(doc, "2.2. VR-TTS — Visual Reasoning via Test-Time Scaling", level=2)
    add_runs(doc, [
        ("Ý tưởng: ", {"bold": True}),
        ("Bắt chước \"test-time scaling\" của LLM (o1, DeepSeek-R1) cho VLM — cho model \"nhìn lại\" ảnh nhiều lần (zoom vào vùng attention cao, đánh dấu SoM, tự hỏi sub-question).", {}),
    ])
    add_table(doc,
        headers=["Quy mô", "Δ Accuracy", "Kết luận"],
        rows=[
            ["n = 100, K=2",  "+0.020", "Tương tự MVP v4"],
            ["n = 1000, K=2", "−0.006  (CI95 [−0.019, +0.008])", "Sụp đổ"],
        ],
        col_widths=[3.0, 6.5, 6.5],
    )

    add_heading(doc, "2.3. Bài học rút ra (đóng góp tự thân)", level=2)
    add_para(doc, "Hai phương pháp độc lập, cùng pattern thất bại, dẫn đến một kết luận có giá trị khoa học:")
    add_quote(doc,
        "Các phương pháp training-free không thể thêm thông tin mới vào hệ thống. "
        "Chúng chỉ sắp xếp lại logits trên cùng một substrate thông tin mà LLaVA đã thấy. "
        "Baseline LLaVA-1.5-7B trên Reefknot spatial subset = 0.504 (≈ chance) — model thật sự không có câu trả lời. "
        "Mọi can thiệp training-free đều đụng trần cấu trúc.")
    add_runs(doc, [
        ("Đây là một phát hiện có thể publish riêng", {"bold": True}),
        (" dưới dạng \"negative scaling study\" (hầu hết paper training-free chỉ báo cáo n<500, ít ai test n=1000).", {}),
    ])

    add_heading(doc, "3. Hướng mới: ALVC (Adaptive-Length Vision Connector)", level=1)

    add_heading(doc, "3.1. Lý do chuyển hướng", level=2)
    add_bullet(doc, "Training-free đã chứng minh có trần cứng → cần một hướng có training nhưng vẫn nhẹ và có nền tảng lý thuyết.")
    add_bullet(doc, "Em đã rà soát 10+ ứng viên hướng, cuối cùng chọn ALVC vì nó là một architecture primitive ở tầng connector (cùng \"loại\" với Q-Former của BLIP-2) — đủ tham vọng nhưng vẫn khả thi trong 1-2 GPU-tuần.")

    add_heading(doc, "3.2. Vấn đề ALVC giải quyết", level=2)
    add_para(doc, "Mọi VLM hiện tại (LLaVA, BLIP-2, Phi-3.5-Vision, Qwen-VL, GPT-4o) đều biến ảnh thành số lượng token cố định trước khi đưa vào LLM, bất kể câu hỏi là gì:")
    add_bullet(doc, "Câu hỏi \"có con mèo không?\" → 144 tokens")
    add_bullet(doc, "Câu hỏi \"mô tả chi tiết căn phòng\" → cũng 144 tokens")
    add_para(doc, "→ Lãng phí compute, và về lý thuyết thông tin là không tối ưu.")

    add_heading(doc, "3.3. Ý tưởng ALVC", level=2)
    add_runs(doc, [
        ("Thay connector cố định bằng một ", {}),
        ("policy học được π(K | image, query)", {"bold": True}),
        (" sinh ra ", {}),
        ("K biến thiên", {"bold": True}),
        (" (tùy ảnh + tùy câu hỏi). Câu hỏi đơn giản → K nhỏ; câu hỏi phức tạp → K lớn.", {}),
    ])
    add_runs(doc, [
        ("Theorem hỗ trợ", {"bold": True}),
        (" (sẽ chứng minh trong Phase 1):", {}),
    ])
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("K(x, q)  ≥  ⌈ I(X; Y | Q) / C ⌉")
    run.bold = True
    run.font.size = Pt(13)
    run.font.name = "Cambria Math"
    add_para(doc, "với C = channel capacity liên tục per-token (đơn vị nats). Đây là cận dưới Shannon-style; ALVC là phương án đạt cận này thông qua PonderNet-trained halting.")

    add_heading(doc, "3.4. Vị thế so với SOTA", level=2)
    add_table(doc,
        headers=["Model", "K theo ảnh?", "K theo câu hỏi?", "Halting học được?"],
        rows=[
            ["LLaVA-1.5 / BLIP-2 / Phi-3.5-V",       "Không", "Không",  "Không"],
            ["Qwen2.5-VL, GPT-4o, LLaVA-NeXT",       "Có",   "Không",  "Không"],
            ["Matryoshka M3 (paper 2024)",            "Có (chọn thủ công)", "Không", "Không"],
            ["ALVC (đề xuất)",                        "Có",   "CÓ",     "CÓ"],
        ],
        col_widths=[5.5, 3.2, 3.6, 4.0],
    )
    add_runs(doc, [
        ("→ ", {}),
        ("Whitespace thật:", {"bold": True}),
        (" chưa paper nào làm query-conditional K. Đây là differentiator cốt lõi (gate G5 ở tuần 1).", {}),
    ])

    add_heading(doc, "4. Tình hình ALVC hiện tại (Phase 0 hoàn tất)", level=1)

    add_heading(doc, "4.1. Đã làm", level=2)
    add_bullet(doc, "Khảo sát lý thuyết: 2 agent đối chiếu literature — whitespace sạch (zero paper kết hợp query-conditioned K + PonderNet halting + rate-distortion ở connector VLM).")
    add_bullet_rich(doc, [
        ("Sửa theorem ban đầu", {"bold": True}),
        (" (K ≤ I/log(vocab) là vacuous vì vision embeddings là continuous, không phải discrete tokens) → bản hiện tại dùng channel capacity liên tục.", {}),
    ])
    add_bullet(doc, "Backbone: pivot từ TinyLLaVA (lỗi tương thích) sang Phi-3.5-Vision-Instruct (Microsoft maintained).")
    add_bullet(doc, "Verify thực nghiệm dim của projector = 4096 (không phải 1024 như config nói) — chi tiết đã suýt phá vỡ tính toán shape.")
    add_bullet(doc, "Skeleton 4 module: connector, halting, training, eval — branch feat/alvc-phase0 (5 commits, đã push).")
    add_bullet(doc, "Sanity script chạy forward end-to-end Phi-3.5-Vision trên GPU thành công.")
    add_bullet(doc, "4 unit tests pass.")

    add_heading(doc, "4.2. Đang / sắp làm (Phase 0 — 2 tuần)", level=2)
    add_table(doc,
        headers=["Tuần", "Nội dung"],
        rows=[
            ["Tuần 1", "Implement HaltingHead (PonderNet-style) + ALVCConnector + viết doc theorem"],
            ["Tuần 1", "Test G5 — gate quyết định: chứng minh K thật sự phụ thuộc câu hỏi, không chỉ ảnh (nếu fail → paper sụp về Matryoshka territory)"],
            ["Tuần 2", "G1-G4 (sanity gates) + ablations + design doc cho Phase 1"],
        ],
        col_widths=[2.5, 13.5],
    )

    add_heading(doc, "4.3. Câu hỏi mở", level=2)
    add_bullet(doc, "Cách trích query embedding cho halting head (LM hidden state token cuối query, hay pooled query)?")
    add_bullet(doc, "Differentiable top-K: Gumbel-softmax (variance cao, rẻ) vs expected-loss qua lưới K (đắt, gradient sạch)?")
    add_bullet(doc, "Dataset cho G5: GQA-Balanced có cấu trúc multiple-questions-per-image — đang xác nhận.")

    add_heading(doc, "4.4. Rủi ro chính", level=2)
    add_bullet_rich(doc, [
        ("PonderNet collapse", {"bold": True}),
        (" về K=1 hoặc K=K_max (đã có khung KL-regularization để giảm rủi ro).", {}),
    ])
    add_bullet_rich(doc, [
        ("Matryoshka M3", {"bold": True}),
        (" là threat gần nhất — phải lead paper bằng query-conditional, không phải \"variable K\" chung chung.", {}),
    ])
    add_bullet_rich(doc, [
        ("Time-to-publish", {"bold": True}),
        (": nếu một lab lớn (OpenAI/Google) đã làm nội bộ và sắp public, có thể bị scoop. Cần publish nhanh nếu Phase 0 G5 pass.", {}),
    ])

    add_heading(doc, "5. Đề xuất", level=1)
    add_bullet(doc, "Tiếp tục Phase 0 ALVC trong 2 tuần tới, lấy kết quả gate G5 làm điểm quyết định go/no-go.")
    add_bullet_rich(doc, [
        ("Song song: ", {"bold": True}),
        ("chuẩn bị một short paper về phát hiện \"n=1000 collapse của training-free interventions\" — đã có data đầy đủ, có thể submit workshop trong khi ALVC còn đang phát triển.", {}),
    ])
    add_bullet(doc, "Em mong nhận được phản hồi của thầy/cô về (a) framing query-conditional có đủ làm contribution chính không, (b) có nên đầu tư thời gian cho short paper negative-finding hay tập trung toàn lực cho ALVC.")

    add_hr(doc)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run("Trân trọng,\nPhạm Văn Trường")
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.italic = True

    doc.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    build()

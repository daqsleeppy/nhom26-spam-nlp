"""NHÓM 26 | Web thực nghiệm phân loại Spam tiếng Việt.

Run: streamlit run app.py
The model pipeline and evaluation mirror Nhom26_NLP_CuoiKy_NangCap.ipynb.
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model_core import (
    DATA_URL,
    MODEL_NAMES,
    classify_one,
    get_all_models,
    matching_features,
    pack_results,
    prepare_dataset,
    run_experiment,
)

st.set_page_config(
    page_title="Nhóm 26 | Lab phân loại Spam",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1230px}
.hero {background: linear-gradient(112deg,#11283e,#165264 75%,#0b826f);
  border-radius:22px;color:white;padding:28px 32px;margin-bottom:16px}
.hero h1 {font-size:2.2rem;margin:0 0 8px 0;color:#fff;line-height:1.15}
.hero p {font-size:1rem;color:#d7eef2;margin:0}
.eyebrow {font-size:.73rem;color:#b2f5e8!important;font-weight:800;
  letter-spacing:.13em;text-transform:uppercase;margin-bottom:10px!important}
.result-good {background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46;
  padding:17px;border-radius:14px;font-weight:800;font-size:1.15rem}
.result-bad {background:#fff2f2;border:1px solid #fecaca;color:#9a1726;
  padding:17px;border-radius:14px;font-weight:800;font-size:1.15rem}
.small-note {font-size:.86rem;color:#576e7b}
div[data-testid="stMetric"] {background:#fff;border:1px solid #d9e9ed;
  border-radius:16px;padding:15px 15px}
div[data-testid="stMetricLabel"] p {color:#48616e;font-size:.83rem}
section[data-testid="stSidebar"] {border-right:1px solid #e2ebed}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <p class="eyebrow">NLP-01 · Nhóm 26 · Phòng thực nghiệm</p>
  <h1>Phát hiện Spam tiếng Việt</h1>
  <p>Thử tin nhắn, đối chiếu 4 thuật toán và kiểm tra kết quả từ cùng một quy trình huấn luyện.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_public_dataset():
    return pd.read_csv(DATA_URL, encoding="utf-8-sig")


with st.sidebar:
    st.header("⚙️ Cấu hình thực nghiệm")
    st.caption("Phù hợp để chạy trên máy hoặc Streamlit Community Cloud")
    source = st.radio(
        "Nguồn dữ liệu", ["Dữ liệu SMS công khai (GitHub)", "Tải lên CSV của nhóm"],
        index=0,
    )
    upload = None
    if source == "Tải lên CSV của nhóm":
        upload = st.file_uploader("Tệp .csv có cột message và label (0/1)", type=["csv"])
    st.divider()
    tune = st.checkbox("Tối ưu mô hình bằng GridSearchCV", value=True)
    st.caption("Luôn dùng 5-fold CV để chọn mô hình. Chỉ dùng tập test sau khi chọn xong.")
    run = st.button("🚀 Huấn luyện & đánh giá", type="primary", use_container_width=True)
    if st.button("Xóa kết quả trong phiên", use_container_width=True):
        st.session_state.pop("exp26", None)
        st.session_state.pop("stats26", None)
        st.session_state.pop("source26", None)
        st.rerun()
    st.divider()
    st.markdown("**4 phương pháp**")
    st.caption("① Word TF-IDF + Naive Bayes\n\n② Word TF-IDF + Logistic Regression\n\n③ Word TF-IDF + Linear SVM\n\n④ Character 3–5 gram + Linear SVM")
    st.caption("Random seed = 42 · Train/Test = 80/20")

if run:
    if source == "Tải lên CSV của nhóm" and upload is None:
        st.warning("Hãy tải lên CSV trước khi bấm huấn luyện.")
    else:
        try:
            with st.spinner("Đang đọc dữ liệu..."):
                raw = get_public_dataset() if upload is None else pd.read_csv(
                    io.BytesIO(upload.getvalue()), encoding="utf-8-sig"
                )
                cleaned, stats = prepare_dataset(raw)
            progress_bar = st.progress(0, text="Chuẩn bị huấn luyện...")
            def show_progress(frac, label):
                progress_bar.progress(max(0.0, min(1.0, float(frac))), text=label)
            exp = run_experiment(cleaned, tune_best=tune, progress=show_progress)
            st.session_state["exp26"] = exp
            st.session_state["stats26"] = stats
            st.session_state["source26"] = source
            progress_bar.empty()
            st.success("Huấn luyện xong. Bạn có thể thử tin nhắn và tải kết quả thực nghiệm.")
        except Exception as ex:
            st.error(f"Chưa thể chạy thực nghiệm: {ex}")
            if source.startswith("Dữ liệu SMS"):
                st.info("Nếu máy/chỗ triển khai không truy cập được GitHub, chọn 'Tải lên CSV của nhóm' và đưa vào `full_dataset.csv`.")

if "exp26" not in st.session_state:
    st.info("**Bắt đầu:** trong thanh bên trái, chọn nguồn dữ liệu rồi bấm **Huấn luyện & đánh giá** để tạo mô hình thật từ CSV.")
    a, b, c = st.columns(3)
    a.metric("Thuật toán", "04")
    b.metric("Đánh giá", "5-fold CV")
    c.metric("Tỷ lệ train/test", "80 / 20")
    st.markdown("### Web có thể làm gì?")
    st.markdown("""
- **Thử 1 tin nhắn:** xem cả bốn mô hình dự đoán Bình thường hay Spam và giải thích các đặc trưng đóng góp.
- **Đánh giá thực nghiệm:** xem Precision, Recall, F1 Spam, Accuracy và ma trận nhầm lẫn.
- **Phân tích lỗi:** xem những tin nhắn mô hình dự đoán sai (FN/FP).
- **Kiểm tra hàng loạt:** tải CSV đầu vào và tải bảng kết quả về máy.
""")
    st.caption("Dữ liệu công khai: Vietnamese SMS Dataset trên GitHub. Nhóm chưa tích hợp PhoBERT vào bản web này.")
    st.stop()

exp = st.session_state["exp26"]
stats = st.session_state["stats26"]
all_models = get_all_models(exp)
st.caption(f"Bộ dữ liệu: {st.session_state['source26']} · {stats['cleaned']:,} tin nhắn hợp lệ · Train: {exp['train_size']:,} · Test: {exp['test_size']:,}")

best_metric_row = exp["test_table"].loc[exp["test_table"]["Mô hình"] == exp["final_name"]].iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy · Final", f"{best_metric_row['Accuracy']:.1%}")
col2.metric("F1 Spam · Final", f"{best_metric_row['F1 Spam']:.1%}")
col3.metric("Recall Spam · Final", f"{best_metric_row['Recall Spam']:.1%}")
col4.metric("Đã làm sạch", f"{stats['cleaned']:,} mẫu")

tab_try, tab_compare, tab_errors, tab_batch, tab_guide = st.tabs([
    "🔎 Thử tin nhắn", "📊 So sánh mô hình", "🧩 Phân tích lỗi",
    "📁 Dự đoán hàng loạt", "📚 Quy trình & xuất file"
])

with tab_try:
    st.subheader("Kiểm tra trực tiếp")
    st.write("Nhập một SMS bất kỳ. Cả 4 mô hình được huấn luyện trên cùng dữ liệu sẽ dự đoán độc lập.")
    examples = {
        "Tự nhập": "",
        "Ví dụ · Tin đáng ngờ": "Chúc mừng bạn trúng thưởng. Vui lòng truy cập link để nhận quà ngay.",
        "Ví dụ · Viết không dấu": "Chuc mung ban trung thuong, click link de nhan qua.",
        "Ví dụ · Tin thường": "Tối nay 7 giờ mình gặp nhau ở cổng trường nhé.",
        "Ví dụ · Tin thông báo": "Ngân hàng thông báo giao dịch của bạn đã thực hiện thành công.",
    }
    selected_ex = st.selectbox("Tin nhắn minh họa", list(examples.keys()))
    user_msg = st.text_area(
        "Nội dung tin nhắn", value=examples[selected_ex],
        height=125, placeholder="Nhập nội dung tin nhắn tiếng Việt..."
    )
    chosen = st.selectbox("Mô hình cần giải thích", list(all_models.keys()),
                          index=list(all_models.keys()).index(exp["final_name"]))
    if st.button("Kiểm tra ngay", type="primary"):
        if not user_msg.strip():
            st.warning("Hãy nhập nội dung tin nhắn.")
        else:
            st.markdown("#### Kết quả của các mô hình")
            columns = st.columns(2)
            for i, (name, model) in enumerate(all_models.items()):
                result = classify_one(model, user_msg)
                with columns[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{name}**")
                        outcome = "⚠️ SPAM / LỪA ĐẢO" if result["label"] == 1 else "✅ BÌNH THƯỜNG"
                        css_class = "result-bad" if result["label"] == 1 else "result-good"
                        st.markdown(f'<div class="{css_class}">{outcome}</div>', unsafe_allow_html=True)
                        if result["score"] is not None:
                            if result["score_type"].startswith("Margin"):
                                st.caption(f"{result['score_type']}: {result['score']:+.3f}")
                            else:
                                st.caption(f"{result['score_type']}: {result['score']:.1%}")
            st.markdown("#### Đặc trưng ảnh hưởng đến dự đoán")
            contrib = matching_features(all_models[chosen], user_msg)
            st.caption(f"Đây là mức đóng góp của từ / n-gram trong mô hình **{chosen}**; giá trị dương nghiêng về Spam, âm nghiêng về Bình thường. Không phải giải thích nhân quả.")
            if contrib.empty:
                st.info("Không tìm thấy đặc trưng phù hợp với từ vựng đã học của mô hình.")
            else:
                st.dataframe(contrib, hide_index=True, use_container_width=True)
    st.caption("Các ví dụ chỉ dùng minh họa; chưa có nhãn thật nên không tính vào độ đo thực nghiệm.")

with tab_compare:
    st.subheader("Đánh giá khoa học trên cùng bộ dữ liệu")
    st.caption("Mô hình được chọn bằng F1 Spam trung bình của 5-fold CV trên train; tập test 20% chỉ dùng để đánh giá sau cùng.")
    st.markdown("**Kết quả cross-validation trên train (mean ± std)**")
    cv_show = exp["cv_table"].copy()
    cv_show = cv_show[["Mô hình", "accuracy_mean", "precision_spam_mean", "recall_spam_mean", "f1_spam_mean", "f1_spam_std"]]
    cv_show.columns = ["Mô hình", "Accuracy CV", "Precision Spam CV", "Recall Spam CV", "F1 Spam CV", "Độ lệch chuẩn F1"]
    st.dataframe(cv_show.style.format({col: "{:.2%}" for col in cv_show.columns[1:]}), hide_index=True, use_container_width=True)
    st.success(f"Mô hình chọn theo CV: **{exp['best_name']}**")
    if exp["tune_best"]:
        st.caption(f"GridSearchCV: {exp['best_params']} · F1 Spam trung bình sau tuning: {exp['tuned_cv_f1']:.2%}")
    st.markdown("**Kết quả trên test cố định (không dùng để chọn tham số)**")
    test_show = exp["test_table"].copy()
    st.dataframe(test_show.style.format({col: "{:.2%}" for col in test_show.columns if col != "Mô hình"}), hide_index=True, use_container_width=True)
    plot = test_show.set_index("Mô hình")[["Precision Spam", "Recall Spam", "F1 Spam"]]
    st.bar_chart(plot)
    st.caption("Các phương pháp có thể dùng class_weight khác nhau. Tài liệu báo cáo cần mô tả rõ cấu hình từng mô hình.")

with tab_errors:
    st.subheader("Ma trận nhầm lẫn của mô hình cuối cùng")
    left, right = st.columns([1.3, 1])
    with left:
        confusion = exp["confusion"]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        im = ax.imshow(confusion, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1], ["Bình thường", "Spam"])
        ax.set_yticks([0, 1], ["Bình thường", "Spam"])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Nhãn thật")
        for (i, j), value in np.ndenumerate(confusion):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=15,
                    color="white" if value > confusion.max() / 2 else "#102b42")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with right:
        tn, fp, fn, tp = exp["confusion"].ravel()
        st.metric("TP · Phát hiện đúng Spam", int(tp))
        st.metric("FN · Bỏ sót Spam", int(fn))
        st.metric("FP · Báo nhầm tin thường", int(fp))
        st.metric("TN · Nhận đúng tin thường", int(tn))
    st.markdown("**Ví dụ những dự đoán sai trên tập test**")
    st.warning("Nếu dùng CSV riêng chứa tin nhắn thật, hãy ẩn thông tin cá nhân trước khi công bố bảng này.")
    errors = exp["errors"]
    if errors.empty:
        st.info("Không có mẫu phân loại sai trong tập test hiện tại.")
    else:
        typ = st.selectbox("Loại lỗi", ["Tất cả", "FN · Bỏ sót Spam", "FP · Báo nhầm"])
        error_shown = errors if typ == "Tất cả" else errors.loc[errors["error_type"] == typ]
        st.dataframe(error_shown.head(50), hide_index=True, use_container_width=True)
        st.caption(f"Đang xem {min(50,len(error_shown))} / {len(error_shown)} trường hợp.")

with tab_batch:
    st.subheader("Dự đoán nhiều tin nhắn bằng CSV")
    st.write("CSV đầu vào chỉ cần cột **message**. Nếu có nhãn `label` trong CSV, web giữ nguyên để bạn đối chiếu thủ công.")
    sample_file = (Path(__file__).parent / "sample_input.csv").read_bytes()
    st.download_button("Tải CSV mẫu", data=sample_file, file_name="sample_input.csv", mime="text/csv")
    batch_file = st.file_uploader("Chọn file CSV dự đoán", type=["csv"], key="batch_csv")
    if batch_file is not None:
        try:
            batch = pd.read_csv(io.BytesIO(batch_file.getvalue()), encoding="utf-8-sig")
            if "message" not in batch.columns:
                st.error("Không có cột 'message'.")
            elif len(batch) > 5000:
                st.warning("Giới hạn 5.000 tin nhắn mỗi lượt để đảm bảo hiệu năng.")
            else:
                texts = batch["message"].fillna("").astype(str).tolist()
                predictions = batch.copy()
                predictions["message"] = texts
                for name, model in all_models.items():
                    short = {
                        MODEL_NAMES[0]: "NaiveBayes", MODEL_NAMES[1]: "LogisticRegression",
                        MODEL_NAMES[2]: "LinearSVM", MODEL_NAMES[3]: "CharSVM",
                        "FINAL (CV + tuning)": "FinalTuned",
                    }.get(name, name)
                    valid = np.array([bool(s.strip()) for s in texts])
                    pred_values = np.full(len(texts), -1, dtype=int)
                    if valid.any():
                        pred_values[valid] = model.predict(np.array(texts, dtype=object)[valid])
                    predictions[short + "_predict"] = pred_values
                st.dataframe(predictions.head(100), use_container_width=True, hide_index=True)
                st.caption("-1 = tin nhắn rỗng, không phân loại.")
                st.download_button(
                    "⬇️ Tải kết quả phân loại CSV", data=predictions.to_csv(index=False).encode("utf-8-sig"),
                    file_name="nhom26_batch_predictions.csv", mime="text/csv"
                )
        except Exception as ex:
            st.error(f"Không đọc được file CSV: {ex}")

with tab_guide:
    st.subheader("Thông tin thí nghiệm để đưa vào báo cáo")
    x, y, z = st.columns(3)
    x.metric("Dữ liệu gốc", stats["raw"])
    y.metric("Sau làm sạch", stats["cleaned"])
    z.metric("Spam / Bình thường", f"{stats['spam']} / {stats['ham']}")
    st.markdown("**Những bước xử lý được thực hiện**")
    st.markdown(f"""
- Loại bản ghi rỗng, thiếu nhãn hoặc nhãn không thuộc 0/1: **{stats['invalid_removed']} dòng**.
- Nội dung mang nhãn mâu thuẫn: **{stats['conflict_messages']} nội dung** (loại {stats['conflict_rows_removed']} dòng).
- Loại tin nhắn trùng nhau hoàn toàn: **{stats['duplicates_removed']} dòng**.
- Chia train/test **80/20**, `stratify`, `random_state=42`; TF-IDF **chỉ fit trên train**.
- Cross-validation **5 folds trên tập train**; chọn theo F1 lớp Spam.
- {'Có' if exp['tune_best'] else 'Không'} GridSearchCV trên train cho mô hình được chọn.
""")
    st.markdown("**Tải mô hình và kết quả để cập nhật báo cáo / sử dụng trên Colab**")
    st.download_button(
        "⬇️ Tải mô hình và các bảng số liệu (.zip)",
        data=pack_results(exp, stats), file_name="nhom26_ket_qua_va_mo_hinh.zip",
        mime="application/zip", use_container_width=True,
    )
    st.caption("ZIP gồm bảng CV, bảng test, ma trận nhầm lẫn, metadata và 2 mô hình joblib. Không bao gồm tin nhắn gốc.")
    st.info("Lưu ý: các giá trị hiện ra được tính từ dữ liệu vừa huấn luyện trong phiên web, không phải số liệu giả định. Không so trực tiếp với bảng giữa kỳ nếu đã thay đổi dữ liệu/cách chia.")

st.divider()
st.caption("Nhóm 26 · Phân loại SMS tiếng Việt · Chỉ phục vụ nghiên cứu/học tập. Kết quả không khẳng định một tin nhắn chắc chắn an toàn hay lừa đảo.")

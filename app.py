import streamlit as st
from ultralytics import YOLO
from PIL import Image

# optional: torchのエラー抑止用
try:
    import asyncio
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# モデルの読み込み
try:
    model = YOLO("yolo11n.pt")
except Exception as e:
    st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
    st.stop()

st.title("YOLO 物体検出")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="アップロードされた画像", use_container_width=True)
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        st.stop()

    if st.button("物体検出を実行"):
        with st.spinner("推論中..."):
            try:
                results = model(image)

                result_image = results[0].plot()
                result_pil = Image.fromarray(result_image)
                st.image(result_pil, caption="検出結果", use_container_width=True)

                names = results[0].names
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                detected_classes = [names[c] for c in classes]
                st.markdown("### 検出された物体:")
                st.write(list(set(detected_classes)))

            except Exception as e:
                st.error(f"物体検出中にエラーが発生しました: {e}")

# ========================
# 打包兼容修复（必须放在最顶部）
# ========================
import sys
import os

# 1. 彻底禁用 PyTorch 易出错模块（解决 name 未定义/导入报错）
os.environ["PYTORCH_DISABLE_DYNAMO"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_NUMPY_DISABLE"] = "1"

# 2. 提前初始化 torch 核心变量，补全缺失的全局变量
import torch

torch.__name__ = "torch"
torch.__file__ = "torch"  # 手动补全 __file__ 变量，避免运行时缺失

# 3. 提前导入 numpy（避免 PyTorch 动态加载报错）
import numpy as np

# 4. Streamlit 打包兼容（禁用版本检查，修复路径）
if getattr(sys, 'frozen', False):
    os.environ['STREAMLIT_DISABLE_VERSION_CHECK'] = 'true'
    sys.path.insert(0, os.path.dirname(sys.executable))
    sys.path.insert(0, os.path.join(os.path.dirname(sys.executable), "_internal"))


# ========================
# 路径适配函数
# ========================
def get_resource_path(relative_path):
    """获取打包后/开发环境的资源绝对路径"""
    if hasattr(sys, '_MEIPASS'):
        # 打包后，资源在临时目录_MEIPASS
        base_path = sys._MEIPASS
    else:
        # 开发环境，资源在项目根目录
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# ========================
# 核心依赖导入（移除 fonts 相关，避免路径报错）
# ========================
from openai import OpenAI
import streamlit as st
import cv2
import threading
import traceback
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pyttsx3
import tempfile
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization


# ========================
# 直接定义 SimpleCNN 模型（不再依赖 train.py）
# 替代 from train import SimpleCNN
# ========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 适配 48x48 输入尺寸
        self.fc2 = nn.Linear(128, 7)  # 7 种情绪分类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ========================
# 2. 会话状态初始化 - Streamlit会话存储
# ========================
if 'upload_count' not in st.session_state:
    st.session_state.upload_count = 0
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'teenager_profiles' not in st.session_state:
    st.session_state.teenager_profiles = {}
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False
if 'current_teenager_id' not in st.session_state:
    st.session_state.current_teenager_id = None
if 'current_teenager_name' not in st.session_state:
    st.session_state.current_teenager_name = None
if 'ai_chat_history' not in st.session_state:
    st.session_state.ai_chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# ========================
# 3. 系统消息配置 - AI对话基础设置
# ========================
system_messages = [
    {"role": "system",
     "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"}
]

# ========================
# 4. 模型加载与初始化 - 情绪识别模型
# ========================
model_path = get_resource_path(os.path.join('models', 'trained_model.pth'))
pytorch_model = None

try:
    # 手动注册模型相关类，替代从 train.py 导入
    torch.serialization.add_safe_globals([
        SimpleCNN,
        nn.Conv2d,
        nn.ReLU,
        nn.MaxPool2d,
        nn.Linear
    ])
    pytorch_model = torch.load(model_path, weights_only=False)
    pytorch_model.eval()
except FileNotFoundError:
    st.error("模型文件未找到，请检查路径：" + model_path)
    st.stop()
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# ========================
# 5. 情绪识别配置 - 人脸检测和情绪标签
# ========================
face_cascade = cv2.CascadeClassifier(
    get_resource_path(os.path.join('data', 'haarcascades', 'haarcascade_frontalface_default.xml')))
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ========================
# 6. 情绪关怀配置 - 正负情绪响应策略
# ========================
negative_emotion_solutions = {
    'Angry': {
        '方案': '深呼吸，数到10，然后慢慢呼气。尝试做一些放松的运动，如瑜伽。',
        '视频推荐': 'https://www.bilibili.com/video/BV1FQ4y1S7mZ/?spm_id_from=333.337.search-card.all.click&vd_source=42ac7273e02988baa01b05cc7287c50c'
    },
    'Disgust': {
        '方案': '转移注意力，想一些美好的事情。可以闻一些喜欢的香味，如薰衣草香。',
        '视频推荐': 'https://www.bilibili.com/video/BV1tZ4y1r7cR/?spm_id_from=333.337.search-card.all.click&vd_source=42ac7273e02988baa01b05cc7287c50c'
    },
    'Fear': {
        '方案': '找一个安全的地方，和信任的人交流。进行积极的自我暗示。',
        '视频推荐': 'https://www.bilibili.com/video/BV1QK421s79h/?spm_id_from=333.337.search-card.all.click&vd_source=42ac7273e02988baa01b05cc7287c50c'
    },
    'Sad': {
        '方案': '看一部喜剧电影。和朋友出去散步。',
        '视频推荐': 'https://www.bilibili.com/video/BV1Ag4y187wN/?spm_id_from=333.337.search-card.all.click&vd_source=42ac7273e02988baa01b05cc7287c50c'
    }
}
positive_emotion_encouragements = {
    'Happy': '你看起来心情很不错哦，继续保持呀！生活中的美好都被你发现啦！',
    'Surprise': '哇，这份惊喜太棒啦！希望你以后还有更多这样的时刻！',
    'Neutral': '平静也是一种很好的状态呢，享受当下就好！'
}

ai_responses = {
    'Angry': [
        "别生气啦，气坏了身体可不好。试着深呼吸几次，让自己平静下来。",
        "消消气，生气可解决不了问题哦。去做些能让自己放松的事情吧。"
    ],
    'Disgust': [
        "别让这种不舒服的感觉影响你啦，转移下注意力，想点开心的事儿。",
        "要是觉得难受，就去闻闻喜欢的味道，说不定会好一些。"
    ],
    'Fear': [
        "别害怕，你很安全的。和我聊聊天，把心里的害怕说出来。",
        "给自己一些积极的心理暗示，你有足够的能力面对一切。"
    ],
    'Sad': [
        "别难过啦，生活中总会有不如意的时候。做点让自己开心的事，心情可能会好起来。",
        "出去走走，和朋友聚聚，说不定能让你开心一些。"
    ],
    'Happy': [
        "哇，感觉你现在心情超棒！希望这份快乐能一直陪伴着你。",
        "开心就好，继续享受这份好心情吧！"
    ],
    'Surprise': [
        "这份惊喜真不错呀！期待你以后还有更多这样的美好瞬间。",
        "哇，被惊喜到啦！保持这份对生活的期待哦。"
    ],
    'Neutral': [
        "平静的状态也挺好的，享受当下的每一刻。",
        "在这平静中，说不定会有新的发现呢。"
    ]
}

# ========================
# 7. Streamlit界面布局 - 系统主界面
# ========================
st.title("青少年人脸识别情绪关怀系统")
st.markdown("""
    <style>
   .stApp {
        background-color: #f4f4f4;
    }
   .stButton button {
        background-color: #007BFF;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

pytorch_accuracy_placeholder = st.sidebar.empty()
pytorch_accuracy_placeholder.title("PyTorch模型评估指标")

st.sidebar.title("上传统计")
st.sidebar.write(f"已上传文件次数: {st.session_state.upload_count}")

# ========================
# 8. 模型训练线程（禁用训练功能，避免依赖 train_pytorch_model）
# ========================
pytorch_accuracy = "模型已加载（训练功能禁用，适配打包）"
pytorch_accuracy_placeholder.write(f"模型状态: {pytorch_accuracy}")


# ========================
# 9. 核心算法 - 人脸情绪处理
# ========================
def process_face(face_roi, pytorch_model, emotion_labels):
    expected_shape = (1, 1, 48, 48)
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = face_roi / 255.0
    mean = 0.5
    std = 0.5
    face_roi = (face_roi - mean) / std
    face_roi = np.expand_dims(face_roi, axis=0)
    face_roi = np.expand_dims(face_roi, axis=-1)
    face_roi_tensor = torch.from_numpy(face_roi).float().permute(0, 3, 1, 2).to(torch.float32)

    # 禁用 CUDA，避免打包后显卡驱动兼容问题
    # if torch.cuda.is_available():
    #     face_roi_tensor = face_roi_tensor.cuda()
    #     pytorch_model = pytorch_model.cuda()

    try:
        emotion_probs = pytorch_model(face_roi_tensor)
        _, emotion_index = torch.max(emotion_probs, 1)
        emotion = emotion_labels[emotion_index.item()]
        confidence = emotion_probs[0][emotion_index].item()
        return emotion, confidence
    except Exception as e:
        print(f"process_face函数出错: {e}")
        print(traceback.format_exc())
        return "错误", 0


# ========================
# 10. 数据存储 - 情绪记录功能
# ========================
def record_emotion(teenager_id, emotion):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    if teenager_id not in st.session_state.teenager_profiles:
        st.session_state.teenager_profiles[teenager_id] = {
            'name': st.session_state.current_teenager_name,
            'records': []
        }
    st.session_state.teenager_profiles[teenager_id]['records'].append({
        'timestamp': timestamp,
        'emotion': emotion
    })


# ========================
# 11. 功能模块 - 图片上传处理
# ========================
def handle_image_upload():
    uploaded_file = st.file_uploader("请上传图片", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.session_state.upload_count += 1
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                st.error("上传的图片格式不支持或文件损坏，请重新上传。")
                return

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                st.warning("图片中未检测到人脸，请重新上传。")
                return

            teenager_id = st.session_state.current_teenager_id
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_roi = gray[y:y + h, x:x + w]
                emotion, confidence = process_face(face_roi, pytorch_model, emotion_labels)
                record_emotion(teenager_id, emotion)

            st.image(img, channels="BGR", caption="识别结果")
            st.write(f"检测到的情绪为：{emotion}（置信度：{confidence:.2f}）")
            if emotion in negative_emotion_solutions:
                st.write(f"缓解方案：{negative_emotion_solutions[emotion]['方案']}")
                st.write(f"推荐视频：[点击观看]({negative_emotion_solutions[emotion]['视频推荐']})")
            elif emotion in positive_emotion_encouragements:
                st.write(positive_emotion_encouragements[emotion])
        except Exception as e:
            st.error(f"处理图片时发生错误: {e}")


# ========================
# 12. 功能模块 - 视频上传处理
# ========================
def handle_video_upload():
    uploaded_video = st.file_uploader("请上传视频", type=["mp4", "avi"])
    if uploaded_video is not None:
        st.session_state.upload_count += 1
        progress_bar = st.progress(0)
        try:
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_video.read())
            cap = cv2.VideoCapture("temp_video.mp4")
            if not cap.isOpened():
                st.error("上传的视频格式不支持或文件损坏，无法打开，请重新上传。")
                return
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            target_frame_rate = 5
            frame_skip = int(frame_rate / target_frame_rate)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            teenager_id = st.session_state.current_teenager_id
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_skip == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            face_roi = gray[y:y + h, x:x + w]
                            emotion, confidence = process_face(face_roi, pytorch_model, emotion_labels)
                            record_emotion(teenager_id, emotion)
                        st.image(frame, channels="BGR", caption="识别结果")
                        st.write(f"当前帧检测到的情绪为：{emotion}（置信度：{confidence:.2f}）")
                        if emotion in negative_emotion_solutions:
                            st.write(f"缓解方案：{negative_emotion_solutions[emotion]['方案']}")
                            st.write(f"推荐视频：[点击观看]({negative_emotion_solutions[emotion]['视频推荐']})")
                        elif emotion in positive_emotion_encouragements:
                            st.write(positive_emotion_encouragements[emotion])
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
            cap.release()
            os.remove("temp_video.mp4")
        except Exception as e:
            st.error(f"处理视频时发生错误: {e}")


# ========================
# 13. 功能模块 - 摄像头实时检测
# ========================
def handle_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("无法打开摄像头，请检查摄像头连接或权限")
        return
    placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        capture_button = st.button("拍照")
    teenager_id = st.session_state.current_teenager_id

    # 限制摄像头循环次数，避免打包后卡死
    max_frames = 1000
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        placeholder.image(frame, channels="BGR", caption="实时识别结果")

        if capture_button:
            st.session_state.captured_image = frame.copy()
            st.success("已拍照")
            gray_captured = cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2GRAY)
            faces_captured = face_cascade.detectMultiScale(gray_captured, scaleFactor=1.1, minNeighbors=5,
                                                           minSize=(30, 30))
            if len(faces_captured) > 0:
                for (x, y, w, h) in faces_captured:
                    face_roi = gray_captured[y:y + h, x:x + w]
                    emotion, confidence = process_face(face_roi, pytorch_model, emotion_labels)
                    record_emotion(teenager_id, emotion)
                st.image(st.session_state.captured_image, channels="BGR", caption="拍摄的图片识别结果")
                st.write(f"拍摄的人脸检测到的情绪为：{emotion}（置信度：{confidence:.2f}）")
                if emotion in negative_emotion_solutions:
                    st.write(f"缓解方案：{negative_emotion_solutions[emotion]['方案']}")
                    st.write(f"推荐视频：[点击观看]({negative_emotion_solutions[emotion]['视频推荐']})")
                elif emotion in positive_emotion_encouragements:
                    st.write(positive_emotion_encouragements[emotion])
            else:
                st.warning("拍摄的图片中未检测到人脸，请重试。")
            capture_button = False  # 防止重复触发
        frame_count += 1
    cap.release()


# ========================
# 14. 功能模块 - 情绪档案查看
# ========================
def view_profile():
    teenager_id = st.session_state.current_teenager_id
    if teenager_id in st.session_state.teenager_profiles:
        profile = st.session_state.teenager_profiles[teenager_id]
        name = profile['name']
        records = profile['records']
        df = pd.DataFrame(records)
        st.write("情绪记录表格")
        st.dataframe(df)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['timestamp'], df['emotion'])
        ax.set_xlabel('时间')
        ax.set_ylabel('情绪')
        ax.set_title('情绪变化趋势图')
        date_form = DateFormatter("%Y-%m-%d %H:%M")
        ax.xaxis.set_major_formatter(date_form)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning(f"未找到青少年{teenager_id}的档案记录。")


# ========================
# 15. AI对话模块 - 聊天历史处理
# ========================
def make_messages(input: str, n: int = 20) -> list[dict]:
    st.session_state.ai_chat_history.append({"role": "user", "content": input})
    new_messages = []
    new_messages.extend(system_messages)
    if len(st.session_state.ai_chat_history) > n:
        st.session_state.ai_chat_history = st.session_state.ai_chat_history[-n:]
    new_messages.extend(st.session_state.ai_chat_history)
    return new_messages


# ========================
# 16. AI对话模块 - API调用
# ========================
def call_api(user_input, chat_history):
    api_key = "sk-sYs7gXDrn2wVynudnVANUtxYyuMzwgccu9NAIWcPqf4LSDx2"
    base_url = "https://api.moonshot.cn/v1"
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    try:
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=make_messages(user_input),
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API请求发生异常: {str(e)}")
        return None


# ========================
# 17. AI对话模块 - 默认响应逻辑
# ========================
def get_default_response(user_input):
    detected_emotion = "Neutral"
    for emotion in emotion_labels:
        if emotion.lower() in user_input.lower():
            detected_emotion = emotion
            break
    return ai_responses[detected_emotion][0]


# ========================
# 18. AI对话模块 - 交互界面
# ========================
def ai_chat():
    st.title("AI虚拟心理陪伴")
    user_input = st.text_input(
        "请输入文字，AI将为你提供帮助：",
        key="ai_chat_input",
        value=st.session_state.user_input,
        on_change=lambda: st.session_state.update(processing=True)
    )

    if user_input.lower() == "结束聊天":
        st.session_state.user_input = ""
        return

    if user_input:
        api_response = call_api(user_input, st.session_state.ai_chat_history)
        response = api_response if api_response else get_default_response(user_input)

        st.session_state.ai_chat_history.append({"role": "user", "content": user_input})
        st.session_state.ai_chat_history.append({"role": "assistant", "content": response})

        st.write(f"AI回复：{response}")

        # 禁用 pyttsx3 语音，避免打包后音频驱动兼容问题（可选恢复）
        # engine = pyttsx3.init()
        # engine.setProperty('rate', 200)
        # engine.setProperty('volume', 1.0)
        # engine.say(response)
        # engine.runAndWait()

        st.subheader("聊天历史")
        for msg in st.session_state.ai_chat_history:
            st.write(f"{'你' if msg['role'] == 'user' else 'AI'}: {msg['content']}")


# ========================
# 19. 登录逻辑 - 用户认证
# ========================
if not st.session_state.is_logged_in:
    st.subheader("用户档案登录")
    teenager_id = st.text_input("请输入青少年ID", "", key="teenager_id_input")
    teenager_name = st.text_input("请输入青少年姓名", "", key="teenager_name_input")
    if st.button("登录"):
        if teenager_id and teenager_name:
            st.session_state.is_logged_in = True
            st.session_state.current_teenager_id = teenager_id
            st.session_state.current_teenager_name = teenager_name
            st.rerun()
        else:
            st.warning("请输入有效的青少年ID和姓名。")
else:
    st.markdown("### 系统操作说明")
    st.write("本系统支持通过以下方式进行青少年人脸情绪识别和关怀：")
    st.write("- **打开图片**：上传包含青少年的本地图片，系统将检测图片中的人脸并识别情绪，记录到档案中。")
    st.write(
        "- **打开视频**：上传包含青少年的本地视频文件，系统会逐帧检测视频中的人脸并显示情绪识别结果，记录到档案中。")
    st.write("- **打开摄像头**：调用本地摄像头，实时检测并识别摄像头画面中的青少年人脸情绪，可拍照记录情绪到档案中。")
    st.write("- **查看档案**：可以查看青少年的历史情绪变化记录和个性化关怀建议。")
    st.write("- **AI虚拟心理陪伴**：与AI进行对话交流，获取即时的心理咨询建议和情感支持。")

    option = st.radio("操作", ["打开图片", "打开视频", "打开摄像头", "查看档案", "AI虚拟心理陪伴", "退出"])
    if option == "打开图片":
        handle_image_upload()
    elif option == "打开视频":
        handle_video_upload()
    elif option == "打开摄像头":
        handle_camera()
    elif option == "查看档案":
        view_profile()
    elif option == "AI虚拟心理陪伴":
        ai_chat()
    elif option == "退出":
        st.session_state.is_logged_in = False
        st.session_state.current_teenager_id = None
        st.session_state.current_teenager_name = None
        st.rerun()
import time
while True:
    time.sleep(1)  # 每秒检测一次，保持程序运行
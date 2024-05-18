import os
import sys
import time
import wave
import tempfile
import threading
import requests
import json
import openai
import torch
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel as whisper
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# A bigger audio buffer gives better accuracy
# but also increases latency in response.
# 表示音频缓冲时间的常量
AUDIO_BUFFER = 5

model_size = input("请输入模型大小（tiny, small, medium, large-v2, large-v3）: ")

# 根据用户输入的模型大小设置路径
path = os.path.join(os.path.dirname(__file__), model_size)

# 确保路径存在
if not os.path.exists(path):
    print(f"指定的路径 {path} 不存在。")
    exit(1)

# 此函数使用 PyAudio 库录制音频，并将其保存为一个临时的 WAV 文件。
# 使用 pyaudio.PyAudio 实例创建一个音频流，通过指定回调函数 callback 来实时写入音频数据到 WAV 文件。
# time.sleep(AUDIO_BUFFER) 会阻塞执行，确保录制足够的音频时间。
# 最后，函数返回保存的 WAV 文件的文件名。
def record_audio(p, device):
    """Record audio from output device and save to temporary WAV file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filename = f.name
        wave_file = wave.open(filename, "wb")
        wave_file.setnchannels(device["maxInputChannels"])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(device["defaultSampleRate"]))

        def callback(in_data, frame_count, time_info, status):
            """Write frames and return PA flag"""
            wave_file.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paInt16,
            channels=device["maxInputChannels"],
            rate=int(device["defaultSampleRate"]),
            frames_per_buffer=pyaudio.get_sample_size(pyaudio.paInt16),
            input=True,
            input_device_index=device["index"],
            stream_callback=callback,
        )

        try:
            time.sleep(AUDIO_BUFFER)  # Blocking execution while playing
        finally:
            stream.stop_stream()
            stream.close()
            wave_file.close()
            # print(f"{filename} saved.")
    return filename

# 此函数使用 Whisper 模型对录制的音频进行转录，并输出转录结果。
def whisper_audio(filename, model):
    """Transcribe audio buffer and display."""
    segments, info = model.transcribe(filename, beam_size=5, language="zh", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))
    os.remove(filename)
    for segment in segments:
        # 调用 translate_text 函数进行翻译
        translated_texts = translate_text([segment.text])
        # 打印翻译后的文本
        for translated_text in translated_texts:
            print(translated_text)
#翻译函数

# 读取配置文件函数
def load_config():
    with open("configg.txt", "r", encoding='utf-8') as file:
        lines = file.readlines()
        api_key = lines[0].strip()
        model_name = lines[1].strip()
        temp = float(lines[2].strip())
        tpc = int(lines[3].strip())
        maxworkers = int(lines[4].strip())
        lan_AtoB = lines[5].strip()
    return api_key, model_name, temp, tpc, maxworkers, lan_AtoB

# 使用配置文件中的信息
api_key, model_name, temp, tpc, maxworkers, lan_AtoB = load_config()

openai.api_key = api_key  # 设置 API 密钥

def translate_text(texts):
    combined_text = "\n".join([f"{t}" for t in texts])
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": """你是一个多语言翻译程序，当要求翻译特定文本时，仅翻译，不添加多余描述和说明。"""},
            {"role": "user", "content": f"{lan_AtoB}且不要添加任何说明：'{texts}'"}
        ],
        temperature=temp,
        max_tokens=4096,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    translated_texts = response.choices[0].message['content'].strip().split('\n')
    return translated_texts


# main 函数是整个脚本的主控制函数。
# 加载 Whisper 模型，选择合适的计算设备（GPU 或 CPU）。
# 获取默认的 WASAPI 输出设备信息，并选择默认的扬声器（输出设备）。
# 使用 PyAudio 开始录制音频，并通过多线程运行 whisper_audio 函数进行音频转录。
def main():
    """Load model record audio and transcribe from default output device."""
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    # model = whisper("large-v3", device=device, compute_type="float16")
    model = whisper(path, device=device, local_files_only=True)

    print("Model loaded.")

    with pyaudio.PyAudio() as pya:
        # Create PyAudio instance via context manager.
        try:
            # Get default WASAPI info
            wasapi_info = pya.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("Looks like WASAPI is not available on the system. Exiting...")
            sys.exit()

        # Get default WASAPI speakers
        default_speakers = pya.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        if not default_speakers["isLoopbackDevice"]:
            for loopback in pya.get_loopback_device_info_generator():
                # Try to find loopback device with same name(and [Loopback suffix]).
                # Unfortunately, this is the most adequate way at the moment.
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print(
                    """
                    Default loopback output device not found.
                    Run `python -m pyaudiowpatch` to check available devices.
                    Exiting...
                    """
                )
                sys.exit()

        print(
            f"Recording from: {default_speakers['name']} ({default_speakers['index']})\n"
        )

        while True:
            filename = record_audio(pya, default_speakers)
            thread = threading.Thread(target=whisper_audio, args=(filename, model))
            thread.start()

main()

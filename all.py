import os
import sys
import time
import wave
import tempfile
import threading
import requests
import json
import openai
from openai import OpenAI
from googletrans import Translator
import torch
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel as whisper
import tkinter as tk
from tkinter import ttk, messagebox
import configparser
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CONFIG_FILE = "config.ini"
AUDIO_BUFFER = 5
# Global variables for selected options
model_size = None
lan = None
device = None
api_key = None
model_name = None
temp = None
lan_AtoB = None
use_microphone = None
translation_service = None
lan1 = None
# Define translation functions
def translate_text_gpt(texts):
    client = OpenAI(api_key = api_key, base_url="https://api.openai.com/v1")
    combined_text = "\n".join([f"{t}" for t in texts])
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": """你是一个多语言翻译程序，当要求翻译特定文本时，仅翻译，不添加多余描述和说明。"""}, 
            {"role": "user", "content": f"{lan_AtoB}且不要添加任何说明：'{combined_text}'"} 
        ],
        temperature=temp,
        max_tokens=4096,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    translated_texts = response.choices[0].message.content.strip().split('\n')
    return translated_texts
def translate_text_deepseek(texts):
    client = OpenAI(api_key = api_key, base_url="https://api.deepseek.com/v1")
    combined_text = "\n".join([f"{t}" for t in texts])
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": """你是一个多语言翻译程序，当要求翻译特定文本时，仅翻译，不添加多余描述和说明。"""}, 
            {"role": "user", "content": f"{lan_AtoB}且不要添加任何说明：'{combined_text}'"} 
        ],
        temperature=temp,
        max_tokens=4096,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    translated_texts = response.choices[0].message.content.strip().split('\n')
    return translated_texts
def translate_text_google(texts):
    from googletrans import Translator
    translator = Translator()
    translated_texts = []
    for text in texts:
        translation = translator.translate(text, dest=lan1)
        translated_texts.append(translation.text)
    return translated_texts
def save_config():
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'model_size': model_size_var.get(),
        'lan': language_var.get(),
        'api_key': api_key_var.get(),
        'model_name': model_name_var.get(),
        'temp': temp_var.get(),
        'lan_AtoB': lan_AtoB_var.get(),
        'audio_source': audio_source_var.get(),
        'translation_service': translation_service_var.get(),
        'lan1': lan1_var.get()
    }
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)
def load_config():
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
        model_size_var.set(config['DEFAULT'].get('model_size', ''))
        language_var.set(config['DEFAULT'].get('lan', ''))
        api_key_var.set(config['DEFAULT'].get('api_key', ''))
        model_name_var.set(config['DEFAULT'].get('model_name', ''))
        temp_var.set(config['DEFAULT'].get('temp', ''))
        lan_AtoB_var.set(config['DEFAULT'].get('lan_AtoB', ''))
        audio_source_var.set(config['DEFAULT'].get('audio_source', 'System Audio'))
        translation_service_var.set(config['DEFAULT'].get('translation_service', 'gpt'))
        lan1_var.set(config['DEFAULT'].get('lan1', ''))
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
    return filename
def whisper_audio(filename, model):
    """Transcribe audio buffer and display."""
    segments, info = model.transcribe(filename, beam_size=5, language=lan, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))
    os.remove(filename)
    for segment in segments:
        translated_texts = translate_text([segment.text])
        for translated_text in translated_texts:
            output_text.insert(tk.END, translated_text + "\n")
            output_text.see(tk.END)
def translate_text(texts):
    if translation_service == 'gpt':
        return translate_text_gpt(texts)
    elif translation_service == 'deepseek':
        return translate_text_deepseek(texts)
    elif translation_service == 'google':
        return translate_text_google(texts)
def start_recording():
    """Load model, record audio and transcribe from selected device."""
    global model_size, lan, device, api_key, model_name, temp, lan_AtoB, use_microphone, translation_service, lan1
    print("Loading model...")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join(os.path.dirname(__file__), model_size)
    if not os.path.exists(path):
        messagebox.showerror("Error", f"指定的路径 {path} 不存在。")
        return
    try:
        model = whisper(path, device=device_type, local_files_only=True)
    except Exception as e:
        messagebox.showerror("Error", f"模型加载失败: {str(e)}")
        return
    print("Model loaded.")
    with pyaudio.PyAudio() as pya:
        if use_microphone:
            default_device = pya.get_device_info_by_index(pya.get_default_input_device_info()["index"])
        else:
            try:
                wasapi_info = pya.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_device = pya.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                if not default_device["isLoopbackDevice"]:
                    for loopback in pya.get_loopback_device_info_generator():
                        if default_device["name"] in loopback["name"]:
                            default_device = loopback
                            break
                    else:
                        messagebox.showerror("Error", "Default loopback output device not found.")
                        return
            except OSError:
                messagebox.showerror("Error", "Looks like WASAPI is not available on the system.")
                return
        print(f"Recording from: {default_device['name']} ({default_device['index']})\n")
        while True:
            filename = record_audio(pya, default_device)
            thread = threading.Thread(target=whisper_audio, args=(filename, model))
            thread.start()
def start_transcription():
    global model_size, lan, api_key, model_name, temp, lan_AtoB, use_microphone, translation_service, lan1
    model_size = model_size_var.get()
    lan = language_var.get()
    api_key = api_key_var.get()
    model_name = model_name_var.get()
    temp = float(temp_var.get())
    lan_AtoB = lan_AtoB_var.get()
    translation_service = translation_service_var.get()
    lan1 = lan1_var.get()
    use_microphone = audio_source_var.get() == "Microphone"
    if any(param == "" for param in [model_size, lan, api_key, model_name, temp, lan_AtoB, translation_service, lan1]):
        messagebox.showerror("Error", "Please fill in all fields.")
        return
    openai.api_key = api_key
    save_config()  # 添加此行保存配置
    thread = threading.Thread(target=start_recording)
    thread.start()
# GUI Setup
root = tk.Tk()
root.title("实时翻译——Real Time Translation")
# Model selection
ttk.Label(root, text="选择whisper模型:\nSelect Whisper Model Size:").grid(column=0, row=0, padx=10, pady=5, sticky='W')
model_size_var = tk.StringVar()
model_size_combobox = ttk.Combobox(root, textvariable=model_size_var, values=["tiny", "small", "medium", "large-v2", "large-v3"])
model_size_combobox.grid(column=1, row=0, padx=10, pady=5, sticky='W')
# Language selection
ttk.Label(root, text="源语言:\nSource Language:").grid(column=0, row=1, padx=10, pady=5, sticky='W')
language_var = tk.StringVar()
language_combobox = ttk.Combobox(root, textvariable=language_var, values=["zh", "ja", "en"])
language_combobox.grid(column=1, row=1, padx=10, pady=5, sticky='W')
# API Key
ttk.Label(root, text="API Key:").grid(column=0, row=2, padx=10, pady=5, sticky='W')
api_key_var = tk.StringVar()
api_key_entry = ttk.Entry(root, textvariable=api_key_var, show='*')
api_key_entry.grid(column=1, row=2, padx=10, pady=5, sticky='W')
# Model Name (LLMs)
ttk.Label(root, text="大语言模型:\nLLMs:").grid(column=0, row=3, padx=10, pady=5, sticky='W')
model_name_var = tk.StringVar()
model_name_combobox = ttk.Combobox(root, textvariable=model_name_var, values=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", "deepseek-chat"])
model_name_combobox.grid(column=1, row=3, padx=10, pady=5, sticky='W')
# Temperature
ttk.Label(root, text="温度:\nTemperature:").grid(column=0, row=4, padx=10, pady=5, sticky='W')
temp_var = tk.StringVar()
temp_entry = ttk.Entry(root, textvariable=temp_var)
temp_entry.grid(column=1, row=4, padx=10, pady=5, sticky='W')
# Language Translation Request
ttk.Label(root, text="输入翻译要求:\nEnter Translation Request:").grid(column=0, row=5, padx=10, pady=5, sticky='W')
lan_AtoB_var = tk.StringVar()
lan_AtoB_combobox = ttk.Combobox(root, textvariable=lan_AtoB_var, values=[
    "请将日语翻译成中文", 
    "请将英语翻译成中文", 
    "以下の中国語を日本語に翻訳してください", 
    "以下の英語を日本語に翻訳してください", 
    "Please translate the following Chinese into English", 
    "Please translate the following Japanese into English"
])
lan_AtoB_combobox.grid(column=1, row=5, padx=10, pady=5, sticky='W')
# Translation Service selection
ttk.Label(root, text="选择翻译服务:\nSelect Translation Service:").grid(column=0, row=6, padx=10, pady=5, sticky='W')
translation_service_var = tk.StringVar()
translation_service_combobox = ttk.Combobox(root, textvariable=translation_service_var, values=["gpt", "deepseek", "google"])
translation_service_combobox.grid(column=1, row=6, padx=10, pady=5, sticky='W')
# Language Direction for Google Translate
ttk.Label(root, text="Google翻译目标语言:\nGoogle Translate Target Language:").grid(column=0, row=7, padx=10, pady=5, sticky='W')
lan1_var = tk.StringVar()
lan1_combobox = ttk.Combobox(root, textvariable=lan1_var, values=["zh-CN", "zh-TW", "ja", "en"])
lan1_combobox.grid(column=1, row=7, padx=10, pady=5, sticky='W')
# Audio Source
ttk.Label(root, text="选择声音来源:\nSelect Audio Source:").grid(column=0, row=8, padx=10, pady=5, sticky='W')
audio_source_var = tk.StringVar(value="System Audio")
ttk.Radiobutton(root, text="系统音源\nSystem Audio", variable=audio_source_var, value="System Audio").grid(column=1, row=8, padx=10, pady=5, sticky='W')
ttk.Radiobutton(root, text="麦克风\nMicrophone", variable=audio_source_var, value="Microphone").grid(column=2, row=8, padx=10, pady=5, sticky='W')
# Output Text Box
output_frame = ttk.LabelFrame(root, text="翻译结果——Translation Output")
output_frame.grid(column=0, row=9, columnspan=3, padx=10, pady=10, sticky='W')
output_text = tk.Text(output_frame, height=6, wrap=tk.WORD)
output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=output_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
output_text.config(yscrollcommand=scrollbar.set)
# Start Button
ttk.Button(root, text="Start Transcription", command=start_transcription).grid(column=0, row=10, columnspan=3, padx=10, pady=10)
load_config()  # 加载现有配置
root.mainloop()
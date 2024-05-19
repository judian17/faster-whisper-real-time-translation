import os
import sys
import time
import wave
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import torch
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel as whisper
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class WhisperUI:
    def __init__(self, root):
        self.root = root
        root.title("Whisper Audio Transcriber")
        self.root.geometry("600x360")
        # 用于排列组件的frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(pady=20)
        # Model size selection
        self.model_size_label = ttk.Label(self.main_frame, text="Model Size:")
        self.model_size_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.model_size_var = tk.StringVar(value="medium")
        self.model_size_combobox = ttk.Combobox(self.main_frame, textvariable=self.model_size_var)
        self.model_size_combobox['values'] = ("tiny", "small", "medium", "large-v2", "large-v3")
        self.model_size_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        # Language selection
        self.language_label = ttk.Label(self.main_frame, text="Language:")
        self.language_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.language_var = tk.StringVar(value="zh")
        self.language_combobox = ttk.Combobox(self.main_frame, textvariable=self.language_var)
        self.language_combobox['values'] = ("zh", "ja", "en")
        self.language_combobox.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        # Compute type selection
        self.compute_type_label = ttk.Label(self.main_frame, text="Compute Type:")
        self.compute_type_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.compute_type_var = tk.StringVar(value="int8")
        self.compute_type_combobox = ttk.Combobox(self.main_frame, textvariable=self.compute_type_var)
        self.compute_type_combobox['values'] = ("int8", "float16")
        self.compute_type_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        # bs selection
        self.bs_label = ttk.Label(self.main_frame, text="Beam Size:")
        self.bs_label.grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.bs_var = tk.StringVar(value="3")
        self.bs_combobox = ttk.Combobox(self.main_frame, textvariable=self.bs_var)
        self.bs_combobox['values'] = ("1", "2", "3", "4", "5")
        self.bs_combobox.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # Buffer size input
        self.buffer_size_label = ttk.Label(self.main_frame, text="Audio Buffer Size (seconds):")
        self.buffer_size_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.buffer_size_entry = ttk.Entry(self.main_frame)
        self.buffer_size_entry.insert(0, "5")
        self.buffer_size_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        # msdm input
        self.msdm_label = ttk.Label(self.main_frame, text="msdm:")
        self.msdm_label.grid(row=2, column=2, padx=5, pady=5, sticky="e")
        self.msdm_entry = ttk.Entry(self.main_frame)
        self.msdm_entry.insert(0, "600")
        self.msdm_entry.grid(row=2, column=3, padx=5, pady=5, sticky="w")
        # Start button
        self.start_button = ttk.Button(self.main_frame, text="Start Transcribing", command=self.start_transcribing)
        self.start_button.grid(row=3, column=1, columnspan=2, pady=10)
        # Output display
        self.output_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
        self.output_display.pack(pady=10)
        self.is_running = False
    def start_transcribing(self):
        if self.is_running:
            return
        self.is_running = True
        threading.Thread(target=self.run_transcriber).start()
    def run_transcriber(self):
        model_size = self.model_size_var.get()
        lan = self.language_var.get()
        computesize = self.compute_type_var.get()
        AUDIO_BUFFER = float(self.buffer_size_entry.get())
        msdm = int(self.msdm_entry.get())
        bs = int(self.bs_var.get())
        # Path based on selected model size
        path = os.path.join(os.path.dirname(__file__), model_size)
        # Ensure the path exists
        if not os.path.exists(path):
            self.output_display.insert(tk.END, f"Specified path {path} does not exist.\n")
            self.is_running = False
            return
        # Load model
        self.output_display.insert(tk.END, "Loading model...\n")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_display.insert(tk.END, f"Using {device} device.\n")
        model = whisper(path, device=device, local_files_only=True)
        self.output_display.insert(tk.END, "Model loaded.\n")
        with pyaudio.PyAudio() as pya:
            try:
                wasapi_info = pya.get_host_api_info_by_type(pyaudio.paWASAPI)
            except OSError:
                self.output_display.insert(tk.END, "WASAPI is not available on the system. Exiting...\n")
                self.is_running = False
                return
            default_speakers = pya.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            if not default_speakers["isLoopbackDevice"]:
                for loopback in pya.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break
                else:
                    self.output_display.insert(tk.END, "Default loopback output device not found. Exiting...\n")
                    self.is_running = False
                    return
            self.output_display.insert(tk.END, f"Recording from: {default_speakers['name']} ({default_speakers['index']})\n")
            while self.is_running:
                filename = self.record_audio(pya, default_speakers, AUDIO_BUFFER)
                threading.Thread(target=self.whisper_audio, args=(filename, model, lan, bs, msdm)).start()


    def record_audio(self, p, device, AUDIO_BUFFER):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            filename = f.name
            wave_file = wave.open(filename, "wb")
            wave_file.setnchannels(device["maxInputChannels"])
            wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wave_file.setframerate(int(device["defaultSampleRate"]))
            def callback(in_data, frame_count, time_info, status):
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
                time.sleep(AUDIO_BUFFER)
            finally:
                stream.stop_stream()
                stream.close()
                wave_file.close()
        return filename
    def whisper_audio(self, filename, model, lan, bs, msdm):
        segments, info = model.transcribe(filename, beam_size=bs, language=lan, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=msdm))
        os.remove(filename)
        for segment in segments:
            self.output_display.insert(tk.END, f"{segment.text}\n")
            self.output_display.see(tk.END)
def main():
    root = tk.Tk()
    app = WhisperUI(root)
    root.mainloop()
if __name__ == "__main__":
    main()

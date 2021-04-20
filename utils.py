import os
import datetime
import gdown
from shutil import unpack_archive
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import subprocess
from pyAudioAnalysis.audioSegmentation import silence_removal
from pyAudioAnalysis.audioBasicIO import read_audio_file
from scipy.io import wavfile


class STTPipeline:
    def __init__(self, m_path):
        self.check_download_models(m_path)
        STT_MODEL_PATH, STT_VOCAB_FILE = self.get_model_files_dirs(m_path)
        self.SAMPLING_RATE = 16000

        print("Initializing STT Model")
        tokenizer = Wav2Vec2CTCTokenizer(STT_VOCAB_FILE, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=self.SAMPLING_RATE, padding_value=0.0,
                                                     do_normalize=True, return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.model = torch.jit.load(STT_MODEL_PATH)

    def get_model_files_dirs(self, models_dir):
        model_dir = os.path.join(models_dir, "stt_model")
        STT_MODEL_PATH = os.path.join(model_dir, "wav2vec_traced_quantized.pt")
        STT_VOCAB_FILE = os.path.join(model_dir, "vocab.json")
        return STT_MODEL_PATH, STT_VOCAB_FILE

    def check_download_models(self, models_dir):
        # set output dir

        model, vocab = self.get_model_files_dirs(models_dir)
        if os.path.exists(model) and os.path.exists(vocab):
            return

        # download and extract
        print("Downloading STT Model")
        op_file = os.path.join(models_dir, "w2v2-53.tar.gz")
        os.makedirs(models_dir, exist_ok=True)
        gdown.download(
            f"https://drive.google.com/uc?id=1m6QXhMF9Zf6P04Z1D2qFiQjEFo16Vexv",
            op_file
        )
        unpack_archive(op_file, models_dir)

    def __call__(self, audio_path):
        audio_input, sr = librosa.load(audio_path, sr=self.SAMPLING_RATE)
        inputs = self.processor(
            audio_input,
            sampling_rate=self.SAMPLING_RATE,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = self.model(inputs.input_values)['logits']

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription


def video2audio(input_file, audio_file_name):
    command = "ffmpeg -hide_banner -loglevel warning -i \"{}\" -b:a 192k -ac 1 -ar 16000 -vn \"{}\"".format(input_file, audio_file_name)
    try:
        print(f"Running {command}")
        ret = subprocess.call(command, shell=True)
        print("Extracted audio to {}".format(audio_file_name.split("/")[-1]))
    except Exception as e:
        print("Error: ", str(e))
        exit(1)


def extractAudio(input_file, output_dir, smoothing_window = 1.0, weight = 0.1):

    print("Detecting silences...")
    [fs, x] = read_audio_file(input_file)
    segmentLimits = silence_removal(x, fs, 0.05, 0.05, smoothing_window, weight)
    ifile_name = os.path.basename(input_file)

    os.makedirs(output_dir, exist_ok=True)
    files = []

    print("Writing segments...")
    for i, s in enumerate(segmentLimits):
        strOut = "{0:s}_{1:.3f}-{2:.3f}.wav".format(ifile_name, s[0], s[1])
        strOut = os.path.join(output_dir, strOut)
        wavfile.write(strOut, fs, x[int(fs * s[0]):int(fs * s[1])])
        files.append(strOut)

    return files


def process_audio(audio_file, stt: STTPipeline):
    start, end = audio_file.split("/")[-1][:-4].split("_")[-1].split("-")
    transcription = stt(audio_file)
    return start, end, transcription

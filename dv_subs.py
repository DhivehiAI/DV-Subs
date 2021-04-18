import argparse
from utils import *
from tqdm import tqdm
import shutil

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser("DV Subtitler")
    arg_parse.add_argument("input", help="Input file name")
    arg_parse.add_argument("output", help="Output file name")
    arg_parse.add_argument("--model_dir", default="./models", help="STT model files directory")
    arg_parse.add_argument("--temp_dir", default="./temp", help="Temp files directory")

    args = arg_parse.parse_args()

    # Extract audio and generate segments
    files = extractAudio(args.input, args.temp_dir)

    # Init the stt
    stt = STTPipeline(args.model_dir)

    # Start transcribing
    print("Transcribing...")
    with open(args.output, "w") as f:

        for w_file in tqdm(files):
            start, end, transcription = process_audio(w_file, stt)
            if len(transcription.strip()) == 0:
                continue

            write_to_file(f, transcription, 1, (start, end))

    print("Removing temporary files...")
    shutil.rmtree(args.temp_dir)

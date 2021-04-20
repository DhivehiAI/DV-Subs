import argparse
from utils import *
from tqdm import tqdm
import shutil
import srt

def perc_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 2.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("input", help="Input audio file name")
    arg_parse.add_argument("output", help="Output file name")
    arg_parse.add_argument("--model_dir", default="./models", help="STT model files directory")
    arg_parse.add_argument("--temp_dir", default="./temp", help="Temp files directory")
    arg_parse.add_argument("--silence_window", default="1.0", type=perc_float, help="Audio smoothing window")
    arg_parse.add_argument("--silence_weight", default="0.3", type=perc_float, help="Audio silence probabilistic weight")

    args = arg_parse.parse_args()

    # Extract audio and generate segments
    files = extractAudio(
        args.input,
        args.temp_dir,
        smoothing_window=args.silence_window,
        weight=args.silence_weight
    )

    # Init the stt
    stt = STTPipeline(args.model_dir)

    # Start transcribing
    print("Transcribing...")

    subs = []
    for w_file in tqdm(files):
        start, end, transcription = process_audio(w_file, stt)
        if len(transcription.strip()) == 0:
            continue

        subs.append(srt.Subtitle(
            0,
            datetime.timedelta(seconds=float(start)),
            datetime.timedelta(seconds=float(end)),
            transcription
        ))

    # write output
    print(f"Writing subtitle file to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        srt.sort_and_reindex(subs, in_place=True)
        f.write(srt.compose(subs))

    print("Removing temporary files...")
    shutil.rmtree(args.temp_dir)

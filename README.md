## DV Subs - Simple video subtitle generator for Dhivehi

This is a simple demonstration of a use case for an ASR toolchain,
such as the Hugging Face wav2vec2 model mentioned on
[https://dhivehi.ai/docs/technologies/stt/](https://dhivehi.ai/docs/technologies/stt/)

The tutorial is inspired by [this article](https://towardsdatascience.com/generating-subtitles-automatically-using-mozilla-deepspeech-562c633936a7)
published towardsdatascience.com. For more in-depth reading about 
the process, please refer to it. *This demo borrows a lot of code from it.*

Additionally, for a manual walk-through, a tutorial notebook
is included.

The process follows a few basic steps:
 * Extract audio from the video
 * Download STT pretrained model and setup inference pipeline
 * Run STT on the audio to transcribe the audio
 * generate a .srt file containing subtitles with timestamps

### Setup

* Clone the repo
* Install requirements `pip install -r requirements.txt`

### Usage
```shell
usage: DV Subtitler [-h] [--model_dir MODEL_DIR] [--temp_dir TEMP_DIR] input output

positional arguments:
  input                 Input file name
  output                Output file name

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        STT model files directory
  --temp_dir TEMP_DIR   Temp files directory
```

import argparse
from utils import *

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser("Extract Audio")
    arg_parse.add_argument("input", help="Input file name")
    arg_parse.add_argument("output", help="Output file name")

    args = arg_parse.parse_args()

    files = video2audio(args.input, args.output)

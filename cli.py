import argparse
import sys
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

from main import generate_prediction


def main():
    parser = argparse.ArgumentParser(
        description="upscalemp3_v2 — Restore audio quality from MP3-compressed files"
    )
    parser.add_argument(
        "input",
        help="Path to input audio file (mp3, wav, flac, etc.)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "-f", "--output-filename",
        default="output.wav",
        help="Output filename (default: output.wav)",
    )
    parser.add_argument(
        "-m", "--model-dir",
        default=str(BASE / "upscalemp3_v2"),
        help="Directory containing model files (default: upscalemp3_v2/)",
    )
    parser.add_argument(
        "-n", "--model-name",
        default="model_13M",
        help="Model filename without extension (default: model_13M)",
    )
    parser.add_argument(
        "-d", "--clip-duration",
        type=float,
        default=1.0,
        help="Clip duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-w", "--overlap",
        type=float,
        default=0.25,
        help="Window overlap ratio 0.0-1.0 (default: 0.25)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    audio_dir = str(input_path.parent)
    audio_filename = input_path.name

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(BASE / "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input:    {input_path}")
    print(f"Model:    {args.model_dir}/{args.model_name}")
    print(f"Output:   {output_dir}")
    print(f"Overlap:  {args.overlap * 100:.0f}%")
    print()

    generate_prediction(
        model_dir=args.model_dir,
        model_filename=args.model_name,
        audio_dir=audio_dir,
        audio_filename=audio_filename,
        clip_duration_seconds=args.clip_duration,
        window_overlap_ratio=args.overlap,
        output_filename=args.output_filename,
        output_dir=output_dir,
    )

    print(f"\nDone. Output saved to {output_dir}/{args.output_filename}")


if __name__ == "__main__":
    main()

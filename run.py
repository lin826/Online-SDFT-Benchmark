"""Run the Liquid LFM2.5 online-SDFT notification benchmark."""

import argparse
from pathlib import Path

from online_sdft.config import (
    ICL_K,
    MODEL_ID,
    PROMPT_STYLE,
    PROMPT_STYLES,
    RAG_K,
    RAG_TEXT_WEIGHT,
)
from online_sdft.experiment import main
from online_sdft.methods import DEFAULT_RFT_SETTINGS, RFTSettings
from online_sdft.reporting import replot_from_outputs


DEFAULT_OUTPUT_DIR = Path("outputs/bandit")
DEFAULT_FIGURE_DIR = Path("figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default=PROMPT_STYLE)
    parser.add_argument("--icl-examples", type=int, default=ICL_K)
    parser.add_argument("--rag-examples", type=int, default=RAG_K)
    parser.add_argument(
        "--rag-text-weight",
        type=float,
        default=RAG_TEXT_WEIGHT,
        help="RAG blend weight for visible title/body token similarity",
    )
    parser.add_argument(
        "--rft-sampling-temperature",
        type=float,
        default=DEFAULT_RFT_SETTINGS.sampling_temperature,
        help="temperature applied only to the RFT categorical proposal",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="redraw figures from stored artifacts instead of running the model",
    )
    args = parser.parse_args()
    if args.figures_only:
        replot_from_outputs(
            args.output_dir or DEFAULT_OUTPUT_DIR,
            args.figure_dir or DEFAULT_FIGURE_DIR,
        )
    else:
        main(
            seeds=args.seeds,
            model_id=args.model_id,
            device=args.device,
            local_files_only=args.local_files_only,
            seed_start=args.seed_start,
            output_dir=args.output_dir,
            figure_dir=args.figure_dir,
            prompt_style=args.prompt_style,
            icl_examples=args.icl_examples,
            rag_examples=args.rag_examples,
            rag_text_weight=args.rag_text_weight,
            rft_settings=RFTSettings(
                sampling_temperature=args.rft_sampling_temperature,
            ),
        )

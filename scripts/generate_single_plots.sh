mkdir -p single_plots
    ### Vision
    # "pope_pope-full/accuracy__POPE-final-Accuracy": "pope",
    # "ocid-ref_ocid-ref-full/accuracy__OCIDRef-All": "OCIDRe",
    # "refcoco_refcoco-full/accuracy__RefCOCO": "RefCOCO",
    # "text-vqa_text-vqa-full/accuracy__TextVQA-Pure": "TextVQA",
    # "vizwiz_vizwiz-full/accuracy__VizWiz-Overall": "VizWiz",
    # "gqa_gqa-full/accuracy": "gqa",
    # "vqa-v2_vqa-v2-full/accuracy": "vqa-v2",
    # "z-score/value": "z-score",
    # "stable-score/value": "stable-score"

    ### Text
    # "results/agieval_sat_en/acc,none": "agieval_sat_en",
    # "results/arc_easy/acc,none": "arc_easy",
    # "results/bigbench_conceptual_combinations_multiple_choice/acc,none": "bigbench_conceptual_combinations",
    # "results/bigbench_cs_algorithms_multiple_choice/acc,none": "bigbench_cs_algorithms",
    # "results/boolq/acc,none": "boolq",
    # "results/copa/acc,none": "copa",
    # "results/hellaswag/acc,none": "hellaswag",
    # "results/mathqa/acc,none": "mathqa",
    # "results/piqa/acc,none": "piqa",
    # "results/pubmedqa/acc,none": "pubmedqa",
    # "z-score/value": "z-score",
    # "stable-score/value": "stable-score"

python scripts/plot_bucket_grid.py --dir 1b --ratio 0p1 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/" --vision_key stable-score/value --text_key stable-score/value
python scripts/plot_bucket_grid.py --dir 1b --ratio 0p1 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value

python scripts/plot_epochs_grid.py --dir 1b --bucket 5 --ratio 0p1 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value
python scripts/plot_epochs_grid.py --dir 1b --bucket 1 --ratio 0p1 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value

python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 5 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value
python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 5 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value
python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 1 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value
python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 1 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value

python scripts/plot_bucket_grid_nollava.py --dir 1b --ratio 0p1 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/" --vision_key stable-score/value --text_key stable-score/value &
python scripts/plot_bucket_grid_nollava.py --dir 1b --ratio 0p1 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value &

python scripts/plot_epochs_grid_nollava.py --dir 1b --bucket 5 --ratio 0p1 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value &
python scripts/plot_epochs_grid_nollava.py --dir 1b --bucket 1 --ratio 0p1 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value &

python scripts/plot_mix_ratio_grid_nollava.py --dir 1b --bucket 5 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value &
python scripts/plot_mix_ratio_grid_nollava.py --dir 1b --bucket 5 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value &
python scripts/plot_mix_ratio_grid_nollava.py --dir 1b --bucket 1 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value &
python scripts/plot_mix_ratio_grid_nollava.py --dir 1b --bucket 1 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value &

python scripts/plot_llava_ratio_grid.py --dir 1b --bucket 5 --ratio 0p1 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value &
python scripts/plot_llava_ratio_grid.py --dir 1b --bucket 5 --ratio 0p1 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key stable-score/value --text_key stable-score/value &

python scripts/plot_bucket_grid.py --dir 1b --ratio 0p1 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/" --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_bucket_grid.py --dir 1b --ratio 0p1 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &

python scripts/plot_epochs_grid.py --dir 1b --bucket 5 --ratio 0p1 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_epochs_grid.py --dir 1b --bucket 1 --ratio 0p1 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &

python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 5 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 5 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 1 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 1 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &

python scripts/plot_bucket_grid_nollava.py --dir 1b --ratio 0p1 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/" --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_bucket_grid_nollava.py --dir 1b --ratio 0p1 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &

python scripts/plot_epochs_grid_nollava.py --dir 1b --bucket 5 --ratio 0p1 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_epochs_grid_nollava.py --dir 1b --bucket 1 --ratio 0p1 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &


python scripts/plot_mix_ratio_grid_nollava.py --dir 1b --bucket 5 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_mix_ratio_grid_nollava.py --dir 1b --bucket 5 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_mix_ratio_grid_nollava.py --dir 1b --bucket 1 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_mix_ratio_grid_nollava.py --dir 1b --bucket 1 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &

python scripts/plot_llava_ratio_grid.py --dir 1b --bucket 5 --ratio 0p1 --epoch_num 4 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none &
python scripts/plot_llava_ratio_grid.py --dir 1b --bucket 5 --ratio 0p1 --epoch_num 2 --name_suffix "_1b" --output_dir "single_plots/"  --vision_key vqa-v2_vqa-v2-full/accuracy --text_key results/arc_easy/acc,none


wait
echo "Done"
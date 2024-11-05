mkdir -p paper_plots

python scripts/plot_bucket_grid.py --dir 1b --ratio 0p1 --epoch_num 4 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_bucket_grid.py --dir 1b --ratio 0p1 --epoch_num 2 --name_suffix "_1b" --output_dir "paper_plots/" &

python scripts/plot_epochs_grid.py --dir 1b --bucket 5 --ratio 0p1 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_epochs_grid.py --dir 1b --bucket 1 --ratio 0p1 --name_suffix "_1b" --output_dir "paper_plots/" &

python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 5 --epoch_num 4 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 5 --epoch_num 2 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 1 --epoch_num 4 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_mix_ratio_grid.py --dir 1b --bucket 1 --epoch_num 2 --name_suffix "_1b" --output_dir "paper_plots/" &

python scripts/plot_llava_ratio_grid.py --dir 1b --bucket 5 --ratio 0p1 --epoch_num 4 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_llava_ratio_grid.py --dir 1b --bucket 5 --ratio 0p1 --epoch_num 2 --name_suffix "_1b" --output_dir "paper_plots/" &

python scripts/plot_bucket_grid_nollava.py --dir 1b --ratio 0p1 --epoch_num 4 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_bucket_grid_nollava.py --dir 1b --ratio 0p1 --epoch_num 2 --name_suffix "_1b" --output_dir "paper_plots/" &

python scripts/plot_epochs_grid_nollava.py --dir 1b --bucket 5 --ratio 0p1 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_epochs_grid_nollava.py --dir 1b --bucket 1 --ratio 0p1 --name_suffix "_1b" --output_dir "paper_plots/" &

python scripts/plot_mix_ratio_grid_nollava.py --bucket 5 --epoch_num 4 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_mix_ratio_grid_nollava.py --bucket 5 --epoch_num 2 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_mix_ratio_grid_nollava.py --bucket 1 --epoch_num 4 --name_suffix "_1b" --output_dir "paper_plots/" &
python scripts/plot_mix_ratio_grid_nollava.py --bucket 1 --epoch_num 2 --name_suffix "_1b" --output_dir "paper_plots/"

wait
echo "Done"
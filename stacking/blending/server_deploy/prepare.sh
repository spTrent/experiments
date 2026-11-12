set -eu
cd "$(dirname "$0")"
input_dir=${BLEND_INPUT:-/root/experiments/blend_inputs}
exact_zip=${SESSION6_EXACT_ZIP:-/root/experiments/Session6_exact_20261111.zip}
mkdir -p "$input_dir/session6" "$input_dir/xgb_rich"
unzip -nq "$exact_zip" -d "$input_dir"
unzip -nq Session6_recovered_h4.zip -d "$input_dir"
unzip -nq session6_blend_input.zip -d "$input_dir/session6"
unzip -nq xgb_rich_k8_blend_input.zip -d "$input_dir/xgb_rich"
unzip -nq best_blend_bf2ef3a96ffc.zip -x '__MACOSX/*' -d "$input_dir"
if [ ! -e "$input_dir/blend_best_output" ]; then
    ln -s best_blend_bf2ef3a96ffc "$input_dir/blend_best_output"
fi
for path in 'Session6_exact/ANSWERS_RU.md' 'Session6_recovered_h4/evaluation/h4/model.cbm' 'session6/comparison_meta.json' 'xgb_rich/manifest.json' 'blend_best_output/base_oof.parquet'; do
    test -s "$input_dir/$path"
done

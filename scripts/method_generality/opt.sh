model_name=AutoTimes_Opt_1b

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 2048 \
  --learning_rate 0.001 \
  --itr 1 \
  --train_epochs 10 \
  --use_amp \
  --llm_ckp_dir ./models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62 \
  --gpu 0 \
  --des 'Opt1b' \
  --cosine \
  --tmax 10 \
  --mlp_hidden_dim 256

for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 2048 \
  --learning_rate 0.001 \
  --itr 1 \
  --train_epochs 10 \
  --use_amp \
  --llm_ckp_dir ./models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62 \
  --gpu 0 \
  --des 'Opt1b' \
  --cosine \
  --tmax 10 \
  --mlp_hidden_dim 256 \
  --test_dir long_term_forecast_ETTh1_672_96_AutoTimes_Opt_1b_ETTh1_sl672_ll576_tl96_lr0.001_bt2048_wd0_hd256_hl2_cosTrue_mixFalse_Opt1b_0
done
model_name=AutoTimes_Gpt2

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
  --learning_rate 0.002 \
  --itr 1 \
  --train_epochs 10 \
  --use_amp \
  --llm_ckp_dir ./models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10 \
  --gpu 0 \
  --des 'Gpt2' \
  --cosine \
  --tmax 10 \
  --mlp_hidden_dim 512


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
  --learning_rate 0.002 \
  --itr 1 \
  --train_epochs 10 \
  --use_amp \
  --llm_ckp_dir ./models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10 \
  --gpu 0 \
  --des 'Gpt2' \
  --cosine \
  --tmax 10 \
  --mlp_hidden_dim 512 \
  --test_dir long_term_forecast_ETTh1_672_96_AutoTimes_Gpt2_ETTh1_sl672_ll576_tl96_lr0.002_bt2048_wd0_hd512_hl2_cosTrue_mixFalse_Gpt2_0
done
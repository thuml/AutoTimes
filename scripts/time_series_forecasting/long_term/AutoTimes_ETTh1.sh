model_name=AutoTimes_Llama

# training one model with a context length
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
  --batch_size 256 \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 0 \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last

# testing the model on all forecast lengths
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
  --batch_size 256 \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 0 \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --test_dir long_term_forecast_ETTh1_672_96_AutoTimes_Llama_ETTh1_sl672_ll576_tl96_lr0.0005_bt256_wd0_hd256_hl0_cosTrue_mixTrue_test_0
done

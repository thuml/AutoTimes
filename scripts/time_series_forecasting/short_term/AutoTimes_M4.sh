export CUDA_VISIBLE_DEVICES=1

model_name=AutoTimes_Llama
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --cosine \
  --tmax 10 \
  --weight_decay 0.00001

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.00005 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --cosine \
  --tmax 10 \
  --weight_decay 0.000005

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --learning_rate 0.00005 \
  --des 'Exp' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --cosine \
  --tmax 10 \
  --weight_decay 0.000001

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --cosine \
  --tmax 10

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --weight_decay 0.000005

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --cosine \
  --tmax 10
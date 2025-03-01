# 1-hour-level
python -u main.py --data ETTh1 --features M --input_len 96  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 8 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 10 --train --patience 1 --EMD

python -u main.py --data ETTh2 --features M --input_len 96  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 8 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 10 --train --patience 1 --EMD

python -u main.py --data ECL --features M --input_len 96  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 10 --train --patience 1 --EMD

python -u main.py --data Traffic --features M --input_len 96  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 20 --itr 10 --train --patience 1 --EMD

python -u main.py --data Air --features M --input_len 96  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 10 --train --patience 1 --EMD

python -u main.py --data River --features M --input_len 96  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 10 --train --patience 1 --EMD

# 15-min-level
python -u main.py --data ETTm1 --features M --input_len 384  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 10 --train --patience 1 --EMD

python -u main.py --data ETTm2 --features M --input_len 384  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 10 --train --patience 1 --EMD

python -u main.py --data HomeC --features M --input_len 384  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 10 --train --patience 1 --EMD

# 10-min-level
python -u main.py --data weather --features M --input_len 576  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 10 --train --patience 1 --EMD

python -u main.py --data Solar --features M --input_len 576  --pred_len 96,192,336,720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 10 --train --patience 1 --EMD

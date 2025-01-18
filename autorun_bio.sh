nice -n 19 python main_test4.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 50 \
		            --trial 72 --dataset bioavailability --num_tasks 1 --classification --num_blocks 2 --slstm 0 \
                --data_dir "/home/UWO/ysun2443/code/trimol_dataset/" --num_gc_layers 7 --power 4 --num_dim 128 --num_experts 4 --num_heads 4


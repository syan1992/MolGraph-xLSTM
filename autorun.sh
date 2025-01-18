if [ $1 == 'freesolv' ]
then
	nice -n 19 python main_test4.py --lr_decay_epochs 800 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 300 \
					--trial 3 --dataset freesolv --num_tasks 1 --mlp_layers 1 --data_dir "/home/UWO/ysun2443/code/trimol_dataset/" --num_gc_layers 4  --power 4 --num_dim 64
elif [ $1 == 'delaney' ]
then
	nice -n 19 python main_test4.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 150 \
		            --trial 15 --dataset delaney --num_tasks 1 --mlp_layers 2 --data_dir "/home/UWO/ysun2443/code/trimol_dataset/" --num_gc_layers 4 --power 4 --num_dim 64
	i
elif [ $1 == 'lipophilicity' ]
then
	nice -n 19 python main_test4.py --lr_decay_epochs 800 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 300 \
		            --trial 1 --dataset lipo --num_tasks 1 --mlp_layers 2 --data_dir "/home/UWO/ysun2443/code/trimol_dataset/" --num_gc_layers 4 --power 4 --num_dim 64

elif [ $1 == 'bace' ]
then
	nice -n 19 python main_test4.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 200 \
		            --trial 24 --dataset bace --num_tasks 1 --temp 0.07 --mlp_layers 1 --classification\
			        --wscl 1 --wrecon 0.1 --data_dir "/home/UWO/ysun2443/code/trimol_dataset/" --num_gc_layers 7 --power 4 --num_dim 64
elif [ $1 == 'tox21' ]
then
	nice -n 19 python main_test4.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 100 \
		    --trial 33 --dataset tox21 --num_tasks 12 --temp 0.07 --mlp_layers 1 --classification\
                    --data_dir "/home/UWO/ysun2443/code/trimol_dataset/" --num_gc_layers 7 --power 4 --num_dim 64
elif [ $1 == 'sider' ]
then
	nice -n 19 python main_test4.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 150 \
		            --trial 28 --dataset sider --num_tasks 27 --temp 0.07 --mlp_layers 1 --classification\
                --data_dir "/home/UWO/ysun2443/code/trimol_dataset/" --num_gc_layers 7 --power 4 --num_dim 64
elif [ $1 == 'clintox' ]
then	
	nice -n 19 python main_test4.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 200 \
		            --trial 23 --dataset clintox --num_tasks 2 --temp 0.07 --mlp_layers 2 --classification\
                --data_dir "/home/UWO/ysun2443/code/trimol_dataset/" --num_gc_layers 7 --power 4 --num_dim 64

elif [ $1 == 'bbbp' ]
then
	nice -n 19 python main_test4.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 300 \
			        --trial 18 --dataset bbbp --num_tasks 1 --temp 0.07 --mlp_layers 1 --classification\
			        --data_dir "/home/UWO/ysun2443/code/trimol_dataset/" --num_gc_layers 7 --power 4 --num_dim 64
else
	echo "The input dataset name should be one of thses: 'freesolv', 'delaney', 'lipophilicity', 'bace', 'tox21', 'sider', and 'clintox'."
fi

1. initilize fcn

./init_fcn_script.sh

initilized checkpoint will be at './ckpt_path/FCN_init'


2. training models

./training_single1_script_fcn.sh

training model using polygon targets; checkpoints will be at './ckpt_path/FCN_1'; validation statistics will be at './FCN_statistic_train1'


./training_single2_script_fcn.sh

training model using canny extracted targets; checkpoints will be at './ckpt_path/FCN_2'; validation statistics will be at './FCN_statistic_train2'


./training_multi_script_fcn.sh

training model using both polygon and canny extracted targets; checkpoints will be at './ckpt_path/FCN_multi'; validation statistics will be at './FCN_statistic_multi'

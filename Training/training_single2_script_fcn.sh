
python train_classification_single2.py --crop_height 512 --crop_width 512 --num_val_images 524 --gpu 5 --ckpt ./ckpt_path/FCN_2 --num_epochs 641 --batch_size 8 --validation_step 2 --epoch_start_i 0 --model FCN --dataset_pkl_train /2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512_fix/dataname_train.pkl --dataset_pkl_val /2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512_fix/dataname_valid.pkl --statistic_path ./FCN_statistic_train2



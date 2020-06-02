

python train_classification_multiple.py \
--crop_height 512 \
--crop_width 512 \
--num_val_images 338 \
--gpu 7 \
--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_multi \
--first_ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_init \
--num_epochs 641 \
--batch_size 8 \
--validation_step 2 \
--epoch_start_i 0 \
--model Unet \
--dataset_pkl_train /2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512/dataname_train.pkl \
--dataset_pkl_val /2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512/dataname_valid.pkl \
--statistic_path ./statistic_path/Unet_statistic_multi \
--continue_training False


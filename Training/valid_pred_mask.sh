

#python valid_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 6 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_single1 \
#--model DANet \
#--dataset_pkl_train /2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512/dataname_train.pkl \
#--dataset_pkl_val /2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512/dataname_valid.pkl \
#--post _pred.png

python valid_pred_mask.py \
--crop_height 512 \
--crop_width 512 \
--gpu 6 \
--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_multi \
--model DANet \
--dataset_pkl_train /2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512/dataname_train.pkl \
--dataset_pkl_val /2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512/dataname_valid.pkl \
--post _pred.png

hp_choice_label_copyall_split_trainAndtest.py 
将fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/train_dataset_1_new/split4096_vec_'+str(i)
其中的
imagepath='image_1024'
labelpath='label_1024'
labeldilatpath='label_1024_dilat'
groundtruthpath='groundtruth_1024'
labelerodepath='label_1024_erode'
gterodepath='groundtruth_1024_erode'
分为训练集和验证集
train_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/train_dataset/split4096_vec_'+str(i)
test_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/valid_dataset/split4096_vec_'+str(i)


prepare_hp_dataname_thesis_1024.py：
生成图片大小1024的数据集的dataname的pkl文件。

data_crop.py:
将‘/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset’中的数据集crop为512大小后，写入
'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_512'

prepare_hp_dataname_thesis_512.py：
生成图片大小512的数据集的dataname的pkl文件。


take_groundtruth_loop.py
通过fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/test_dataset_1_new/split4096_vec_' + str(i)
下的
imagepath='image_1024'
labelpath = 'label_1024'
提取gt，生成gtpath='groundtruth_1024'

dilation.py
膨胀
fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/train_dataset_1_new/split4096_vec_'+str(i)
下的
gtpath='label_1024'
生成'label_1024_dilat'
{gt_dilate_path='label_1024_'， os.makedirs(visual_path+'dilat')}

erode.py
腐蚀，代码与dilation.py一致。

erode_gt.py
将    labelpath = 'label_1024_erode'
    groundtruthpath = 'groundtruth_1024'
相乘，生成bulidgtpath = 'groundtruth_1024_erode'




data_crop.py:
将'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/train_dataset_1_new/中的数据集crop为512大小后，写入
'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512'

hp_choice_label_copyall.py:  
从train_dataset_1_new_512生成train_dataset_1_new_512_filter
fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512/split4096_vec_'+str(i)
new_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512_filter/split4096_vec_'+str(i)
挑选连通区域面积大于100的图片,一次将目录下的这些文件，都执行此操作
imagepath='image_512'
labelpath='label_512_dilat'
groundtruthpath='groundtruth_512_erode'


dilation_groundtruth512:
    basepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512_filter'
    labelpath_gt = 'groundtruth_512_erode'
    build_gtdilat = 'groundtruth_512_erode_dilat'


hp_choice_label_copyall_split_trainAndtest.py 
将new_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512_filter/split4096_vec_'+str(i)
其中的
imagepath='image_512'
labelpath='label_512_dilat'
groundtruthpath='groundtruth_512_erode'
groundtruthpath_dilat='groundtruth_512_erode_dilat'
分为训练集和验证集
train_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_512/split4096_vec_'+str(i)
test_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/valid_dataset_512/split4096_vec_'+str(i)


prepare_hp_dataname_thesis_512.py：
生成图片大小512的数据集的dataname的pkl文件。

print(len(x_train_filenames),len(x_test_filenames),len(x_valid_filenames))
3287 772 344



prepare_hp_dataname_thesis_1024.py：
生成图片大小1024的数据集的dataname的pkl文件。


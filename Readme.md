# hp_3
Segmentation of small target HP based on multi label weak supervised learning

Author: 
Yiming Yang, email:minggto@foxmail.com
Yongquan Yang


1.数据集：

数据集处理和生成脚本：/data_code/
（1）take_groundtruth_loop.py
通过fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/test_dataset_1_new/split4096_vec_' + str(i)
下的
imagepath='image_1024'
labelpath = 'label_1024'
提取gt，生成gtpath='groundtruth_1024'

（2）dilation.py
膨胀fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/train_dataset_1_new/split4096_vec_'+str(i)
下的
gtpath='label_1024'
生成'label_1024_dilat'
{gt_dilate_path='label_1024_'， os.makedirs(visual_path+'dilat')}

（3）erode.py
腐蚀，代码与dilation.py一致。

（4）erode_gt.py
将    labelpath = 'label_1024_erode'
    groundtruthpath = 'groundtruth_1024'
相乘，生成bulidgtpath = 'groundtruth_1024_erode'

（5）data_crop.py:
将'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/train_dataset_1_new/中的数据集crop为512大小后，写入
'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512'

（6）hp_choice_label_copyall.py:  
从train_dataset_1_new_512生成train_dataset_1_new_512_filter
fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512/split4096_vec_'+str(i)
new_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512_filter/split4096_vec_'+str(i)
挑选连通区域面积大于100的图片,一次将目录下的这些文件，都执行此操作
imagepath='image_512'
labelpath='label_512_dilat'
groundtruthpath='groundtruth_512_erode'

（7）dilation_groundtruth512.py:
将basepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512_filter'下的
labelpath_gt = 'groundtruth_512_erode'进行膨胀
生成build_gtdilat = 'groundtruth_512_erode_dilat'

（8）hp_choice_label_copyall_split_trainAndtest.py：
将new_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512_filter/split4096_vec_'+str(i)
其中的
imagepath='image_512'
labelpath='label_512_dilat'
groundtruthpath='groundtruth_512_erode'
groundtruthpath_dilat='groundtruth_512_erode_dilat'
分为训练集和验证集
train_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_512/split4096_vec_'+str(i)
test_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/valid_dataset_512/split4096_vec_'+str(i)

（9）prepare_hp_dataname_thesis_512.py：
生成图片大小512的数据集的dataname的pkl文件。


训练集路径
 '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_512'
image、label1和label2目录：'image_512' , 'label_512_dilat' , 'groundtruth_512_erode'
验证集路径
'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/valid_dataset_512'
image、label1和label2目录：'image_512' , 'label_512_dilat' , 'groundtruth_512_erode'
测试集路径
'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512'
image、label1和label2目录：'image_512' , 'label_512' ,  'groundtruth_512'
根据数据集中图像name生成pkl的路径
'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512/'

2.训练：
《新增训练过程中保存的ckpt路径》
ckpt路径:/2_data/yym_workspcae/exp/thesis_hp_3/

3.验证
验证集预测mask：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/Training/
valid_pred_mask.sh
《增加验证预测结果路径》
生成图像在for checkpoint_i in range(441, 641, 2)的ckpt的目录中，
ckpt路径:/2_data/yym_workspcae/exp/thesis_hp_3/
说明：训练时对偶数的ckpt生成了验证集预测mask，这个脚本补充了最后两百个奇数ckpt的验证集预测mask。


在验证集不同ckpt上计算recall和precision：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/val_script/
recall_precision_validation_mutual.py
生成路径：./statistic_path_valid/


根据上一步的recall和precision，选择ckpt：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/val_script/
choose_ckpt.py

run: python val_script/choose_ckpt.py 
#f1
# FCN 127 73 101             -> 568 514 (542)
# Unet 173 21 37             -> 614 462 (478)
# tiny 133 161 113             -> 574 602 (554)
# pspnet 199 95 53             -> (640) 536 494
# DANet 175 131 135             -> 616 572 (576)

##f2
# FCN 173 23 167             -> 614 464 608
# Unet 127 91 135             -> 568 532 576
# tiny 133 59 113             -> (574) 500 (554)
# pspnet 5 47 53             -> 446 488 (494)
# DANet 169 75 33             -> 610 516 474

# ##f3
# FCN 173 23 167             -> (614 464 608)
# Unet 127 16 135             -> (568) 457 (576)
# tiny 173 59 113             -> 614 (500 554)
# pspnet 5 47 53             -> (446 488 494)
# DANet 169 137 33             -> (610) 578 (474)

4.测试
根据选择的ckpt，在测试集上预测mask：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/Training/
test_pred_mask.sh
《增加测试预测结果路径》
生成路径：visual_basepath = '/2_data/share/workspace/yym/HP/hp3_visural/' + model_task

根据测试集上预测mask，没有后处理，直接提取groundtruth（HP）：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/test/
take_groundtruth_loop.sh
《增加中间结果路径》
生成路径：'/2_data/share/workspace/yym/HP/hp3_visural/' + args.model_task + '/split4096_vec_' + str(i)+ '_takegt'

根据测试集上预测mask，增加后处理:选择性膨胀等操作，提取pred_groundtruth（HP）：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/test/
take_post_groundtruth_loop.sh
《增加中间结果路径》
生成路径：'/2_data/share/workspace/yym/HP/hp3_visural/' + args.model_task + '/split4096_vec_' + str(i)+ '_posttakegt'

根据测试集上预测mask，增加后处理:膨胀等操作，提取pred_groundtruth（HP）(实验指标低，不建议采用)：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/test/
take_post1_groundtruth_loop.sh
《增加中间结果路径》
生成路径：'/2_data/share/workspace/yym/HP/hp3_visural/' + args.model_task + '/split4096_vec_' + str(i) + '_post1_takegt'

根据提取的pred_groundtruth与标签groundtruth计算所有测试集上的指标：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/Training/
test_post_metrix_count.sh
《增加中间结果路径》
pkl生成路径：'/2_data/share/workspace/yym/HP/hp3_visural/'+ model_task + '_' + num + '_test_post_metrix_count.pkl'
xls生成路径:'/2_data/share/workspace/yym/HP/hp3_visural/test_post_metrix_count.xls'
说明：这一步对应上面的take_post_groundtruth_loop.sh

test_metrix_count.sh
pkl生成路径：'/2_data/share/workspace/yym/HP/hp3_visural/'+ model_task + '_' + num + '_test_metrix_count.pkl'
xls生成路径:'/2_data/share/workspace/yym/HP/hp3_visural/test_metrix_count.xls'
说明：这一步对应上面的take_groundtruth_loop.sh

test_post1_metrix_count.sh
pkl生成路径：'/2_data/share/workspace/yym/HP/hp3_visural/'+ model_task + '_' + num + '_test_post1_metrix_count.pkl'
xls生成路径:'/2_data/share/workspace/yym/HP/hp3_visural/test_post1_metrix_count.xls'
说明：这一步对应上面的take_post1_groundtruth_loop.sh

5.可视化
在'/2_data/share/workspace/yym/HP/hp3_visural/ShowData/Polygon and Targets/'目录下中形成的可视化图片，少量的：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/val_script/
thesis_hp3_pic.py

在'/2_data/share/workspace/yym/HP/hp3_visural/' + model_task+ '/' + vec + '_visural_target1_target2' + '/' + num的目录下
形成的可视化图片，大量的：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/val_script/
thesis_hp3_show.sh

论文里的配图，生成在targetDir = '/2_data/share/workspace/yym/HP/hp3_visural/show_data/'+ model_task +'/takegt'，配图选择代码：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/val_script/
choose_showpic.py
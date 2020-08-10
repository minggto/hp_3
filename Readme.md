# hp_3
Segmentation of small target HP based on multi label weak supervised learning

Author: 
Yiming Yang, email:minggto@foxmail.com
Yongquan Yang


数据集：
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



测试：

验证集预测mask：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/Training/
valid_pred_mask.sh

在验证集不同ckpt上计算recall和precision：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/val_script/
recall_precision_validation_mutual.py

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




根据选择的ckpt，在测试集上预测mask：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/Training/
test_pred_mask.sh



根据测试集上预测mask，没有后处理，直接提取groundtruth（HP）：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/test/
take_groundtruth_loop.sh

根据测试集上预测mask，增加后处理:选择性膨胀等操作，提取pred_groundtruth（HP）：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/test/
take_post_groundtruth_loop.sh

根据测试集上预测mask，增加后处理:膨胀等操作，提取pred_groundtruth（HP）(实验指标低，不建议采用)：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/test/
take_post1_groundtruth_loop.sh

根据提取的pred_groundtruth与标签groundtruth计算所有测试集上的指标：
/2_data/yym_workspcae/workspace/thesis/hp_3/github/hp_3/Training/
test_post_metrix_count.sh







可视化：

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
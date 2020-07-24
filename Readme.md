# hp_3
Segmentation of small target HP based on multi label weak supervised learning

Author: 
Yiming Yang, email:minggto@foxmail.com
Yongquan Yang


测试：

验证集预测mask：
valid_pred_mask.sh

在验证集不同ckpt上计算recall和precision：
recall_precision_validation_mutual.py

根据上一步的recall和precision，选择ckpt：
choose_ckpt.py

根据选择的ckpt，在测试集上预测mask：
test_pred_mask.sh

根据测试集上预测mask，没有后处理，直接提取groundtruth（HP）：
take_groundtruth_loop.sh

根据测试集上预测mask，增加后处理膨胀等操作，提取pred_groundtruth（HP）：
take_post_groundtruth_loop.sh

根据提取的pred_groundtruth与标签groundtruth计算所有测试集上的指标：
test_post_metrix_count.sh







可视化：

在'/2_data/share/workspace/yym/HP/hp3_visural/ShowData/Polygon and Targets/'目录下中形成的可视化图片，少量的：
thesis_hp3_pic.py

在'/2_data/share/workspace/yym/HP/hp3_visural/' + model_task+ '/' + vec + '_visural_target1_target2' + '/' + num的目录下
形成的可视化图片，大量的：
thesis_hp3_show.sh

论文里的配图，生成在targetDir = '/2_data/share/workspace/yym/HP/hp3_visural/show_data/'+ model_task +'/takegt'，配图选择代码：
choose_showpic.py
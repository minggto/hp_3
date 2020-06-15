import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    image = cv2.imread(path, -1)
    return image

def evaluate_mutual_segmentation(gt1, gt2, pred1, pred2, predm):
    flat_pred1 = pred1.flatten()
    flat_pred2 = pred2.flatten()
    flat_label1 = gt1.flatten()
    flat_label2 = gt2.flatten()
    flat_predm = predm.flatten()
    
    pred_p1 = (flat_pred1==255)
    pred_p2 = (flat_pred2==255)
    pred_n1 = (flat_pred1!=255)
    pred_n2 = (flat_pred2!=255)
    predm_p = (flat_predm==255)
    predm_n = (flat_predm!=255)
    label_p1 = (flat_label1==255)
    label_p2 = (flat_label2==255)
    label_n1 = (flat_label1!=255)
    label_n2 = (flat_label2!=255)
    
    tp11 = np.sum(pred_p1 * label_p1)
    fp11 = np.sum(pred_p1 * label_n1)
    fn11 = np.sum(pred_n1 * label_p1)
    
    tp21 = np.sum(pred_p2 * label_p1)
    fp21 = np.sum(pred_p2 * label_n1)
    fn21 = np.sum(pred_n2 * label_p1)
    
    tpm1 = np.sum(predm_p * label_p1)
    fpm1 = np.sum(predm_p * label_n1)
    fnm1 = np.sum(predm_n * label_p1)
    
    tp22 = np.sum(pred_p2 * label_p2)
    fp22 = np.sum(pred_p2 * label_n2)
    fn22 = np.sum(pred_n2 * label_p2)
    
    tp12 = np.sum(pred_p1 * label_p2)
    fp12 = np.sum(pred_p1 * label_n2)
    fn12 = np.sum(pred_n1 * label_p2)
    
    tpm2 = np.sum(predm_p * label_p2)
    fpm2 = np.sum(predm_p * label_n2)
    fnm2 = np.sum(predm_n * label_p2)
    
    return np.array([tp11, fp11, fn11]), np.array([tp21, fp21, fn21]), np.array([tpm1, fpm1, fnm1]), np.array([tp22, fp22, fn22]), np.array([tp12, fp12, fn12]), np.array([tpm2, fpm2, fnm2])


#ckpt_path1 = '/2_data/share/workspace/yyq/HP-paper-4-tmi/ckpt_path/FCN_1/'
#ckpt_path2 = '/2_data/share/workspace/yym/exp/thesis_hp_3/FCN_single2/'
#ckpt_pathm = '/2_data/share/workspace/yym/exp/thesis_hp_3/FCN_multi/'
#statistic_path = './FCN_12m'

'''
ckpt_path1 = '/2_data/yym_workspcae/exp/thesis_hp_3/Unet_single1/'
ckpt_path2 = '/2_data/yym_workspcae/exp/thesis_hp_3/Unet_single2/'
ckpt_pathm = '/2_data/yym_workspcae/exp/thesis_hp_3/Unet_multi/'
statistic_path = './statistic_path/Unet_statistic_12m1'

ckpt_path1 = '/2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single1/'
ckpt_path2 = '/2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single2/'
ckpt_pathm = '/2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_multi/'
statistic_path = './statistic_path/tiny_deeplabv3_statistic_12m'

ckpt_path1 = '/2_data/yym_workspcae/exp/thesis_hp_3/FCN_single1/'
ckpt_path2 = '/2_data/yym_workspcae/exp/thesis_hp_3/FCN_single2/'
ckpt_pathm = '/2_data/yym_workspcae/exp/thesis_hp_3/FCN_multi/'
statistic_path = './statistic_path/FCN_statistic_12m'

ckpt_path1 = '/2_data/yym_workspcae/exp/thesis_hp_3/pspnet_single1/'
ckpt_path2 = '/2_data/yym_workspcae/exp/thesis_hp_3/pspnet_single2/'
ckpt_pathm = '/2_data/yym_workspcae/exp/thesis_hp_3/pspnet_multi/'
statistic_path = './statistic_path/pspnet_statistic_12m'
'''

ckpt_path1 = '/2_data/yym_workspcae/exp/thesis_hp_3/DANet_single1/'
ckpt_path2 = '/2_data/yym_workspcae/exp/thesis_hp_3/DANet_single2/'
ckpt_pathm = '/2_data/yym_workspcae/exp/thesis_hp_3/DANet_multi/'
statistic_path = './statistic_path/DANet_statistic_12m'


if not os.path.exists(statistic_path):
    os.makedirs(statistic_path)


ckpt_dirs = os.listdir(ckpt_path1)
ckpt_dirs.sort()
print("---ckpt_dirs---",ckpt_dirs)
#exit()

ckpt_dirs = ckpt_dirs[1:641]

valid_ckpt_dirs = []

for cd in ckpt_dirs:
    cdp = os.path.join(ckpt_path1, cd)
    if os.path.isdir(cdp):
        valid_ckpt_dirs.append(cd)
valid_ckpt_dirs.sort()
print(valid_ckpt_dirs)


TFF = []
P1 = []
P21 = []
Pm1 = []
R2 = []
R12 = []
Rm2 = []

for vcd in valid_ckpt_dirs:
    vcdp1 = os.path.join(ckpt_path1, vcd)
    vcdp2 = os.path.join(ckpt_path2, vcd)
    vcdpm = os.path.join(ckpt_pathm, vcd)

    files = os.listdir(vcdp1)
    gt_images = []
    for f in files:
        if '_pred' in f:
            gt_images.append(f)
    if len(gt_images) == 0:
        print("---0---",vcdp1)
        continue

    gtdp1 = os.path.join(ckpt_path1, '0000')
    gtdp2 = os.path.join(ckpt_path2, '0000')
    files = os.listdir(gtdp1)
    gt_images = []
    for f in files:
        if '_gt' in f:
            gt_images.append(f)
    if len(gt_images) == 0:
        print("error--- 0000 no data")
        exit()
    



    print(vcdp1)
    print(vcdp2)
    print(vcdpm)
    
    all_tff11 = np.array([0,0,0])
    all_tff21 = np.array([0,0,0])
    all_tffm1 = np.array([0,0,0])
    all_tff22 = np.array([0,0,0])
    all_tff12 = np.array([0,0,0])
    all_tffm2 = np.array([0,0,0])
    for gi in gt_images:
        gt_img1 = os.path.join(gtdp1, gi)
        gt_img2 = os.path.join(gtdp2, gi)
        pred_img1 = os.path.join(vcdp1, gi.split('gt')[0]+'pred.png')
        pred_img2 = os.path.join(vcdp2, gi.split('gt')[0]+'pred.png')
        pred_imgm = os.path.join(vcdpm, gi.split('gt')[0]+'pred1.png')
        if not os.path.exists(pred_imgm):
            pred_imgm = os.path.join(vcdpm, gi.split('gt')[0] + 'pred.png')
        
        gt1 = load_image(gt_img1)
        gt2 = load_image(gt_img2)
        pred1 = load_image(pred_img1)
        pred2 = load_image(pred_img2)
        predm = load_image(pred_imgm)
        # print('gt1',gt1.shape)
        # print('pred1', pred1.shape)
        # print(pred_imgm)
        # print('predm', predm.shape)
        if len(predm.shape)==3:
            predm = cv2.cvtColor(predm, cv2.COLOR_BGR2GRAY)
            #print('predm',predm.shape)
            
        #exit()
        tff11, tff21, tffm1, tff22, tff12, tffm2 = evaluate_mutual_segmentation(gt1, gt2, pred1, pred2, predm)
    
        all_tff11 += tff11
        all_tff21 += tff21
        all_tffm1 += tffm1
        all_tff22 += tff22
        all_tff12 += tff12
        all_tffm2 += tffm2
    
    tffl11 = all_tff11.tolist()
    tffl21 = all_tff21.tolist()
    tfflm1 = all_tffm1.tolist()
    tffl22 = all_tff22.tolist()
    tffl12 = all_tff12.tolist()
    tfflm2 = all_tffm2.tolist()
    
    print(tffl11)
    print(tffl21)
    print(tfflm1)
    print(tffl22)
    print(tffl12)
    print(tfflm2)
    
    TFF.append([tffl11, tffl21, tfflm1, tffl22, tffl12, tfflm2])

    P1.append(tffl11[1])
    P21.append(tffl21[1])
    Pm1.append(tfflm1[1])
    R2.append(tffl22[0])
    R12.append(tffl12[0])
    Rm2.append(tfflm2[0])


    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(len(P1)), P1, linewidth = '2', label = "precision_task1_target1", color="blue", linestyle=(0, (1, 1)))
    ax1.plot(range(len(P21)), P21, linewidth = '2', label = "precision_task2_target1", color="red", linestyle=(0, (5, 1)))
    ax1.plot(range(len(Pm1)), Pm1, linewidth = '2', label = "precision_task12_target1", color="green", linestyle=(0, (9, 1)))
    ax1.set_title("Precision vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current Precision")
    plt.savefig(os.path.join(statistic_path,'precision_vs_epochs.png'))
    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(len(R12)), R12, linewidth = '2', label = "recall_task1_target2", color="blue", linestyle=(0, (1, 1)))
    ax2.plot(range(len(R2)), R2, linewidth = '2', label = "recall_task2_target2", color="red", linestyle=(0, (5, 1)))
    ax2.plot(range(len(Rm2)), Rm2, linewidth = '2', label = "recall_task12_target2", color="green", linestyle=(0, (9, 1)))
    ax2.set_title("Recall vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current Recall")
    plt.savefig(os.path.join(statistic_path,'recall_vs_epochs.png'))
    plt.clf()

    import pickle
    f=open(os.path.join(statistic_path,'statistics.pckl'), 'wb')
    pickle.dump([TFF, P1, P21, Pm1, R2, R12, Rm2], f)
    f.close()
  
  
        
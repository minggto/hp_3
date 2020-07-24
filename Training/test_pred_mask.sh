
#f1
# FCN 73 173 185             -> 514 614 626
# Unet 15 159 37             -> 456 600 478
# tiny_deeplabv3 177 193 119      -> 618 634 560
# pspnet 138 25 123          -> 579 466 564
# DANet 127 169 187          -> 568 610 628
# add 441
#f2
# FCN 97 173 101            -> 538 614 542
# Unet 15 127 37            -> 456 568 478
# tiny_deeplabv3 109 173 113      -> 550 614 554
# pspnet 138 5 21        -> 579 446 462
# DANet 127 169 185         -> 568 610 626
#f3
# 161 173 101             -> 602 614 542
# 109 127 173             -> 550 568 614
# 109 173 113             -> 550 614 554
# 138 5 21             -> 579 446 462
# 127 169 135             -> 568 610 576

##f1
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_single1 \
#--model FCN \
#--checkpoint_i 514
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_single2 \
#--model FCN \
#--checkpoint_i 614
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_multi \
#--model FCN \
#--checkpoint_i 626

###f2
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_single1 \
#--model FCN \
#--checkpoint_i 538
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_multi \
#--model FCN \
#--checkpoint_i 542
#
#
###f3
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_single1 \
#--model FCN \
#--checkpoint_i 602






##f1
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single1 \
#--model Unet \
#--checkpoint_i 456
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single2 \
#--model Unet \
#--checkpoint_i 600
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_multi \
#--model Unet \
#--checkpoint_i 478


###f2
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single2 \
#--model Unet \
#--checkpoint_i 568
#
###f3
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single1 \
#--model Unet \
#--checkpoint_i 550
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_multi \
#--model Unet \
#--checkpoint_i 614



##f1
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single1 \
#--model tiny_deeplabv3 \
#--checkpoint_i 618
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single2 \
#--model tiny_deeplabv3 \
#--checkpoint_i 634
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_multi \
#--model tiny_deeplabv3 \
#--checkpoint_i 560

###f2
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single1 \
#--model tiny_deeplabv3 \
#--checkpoint_i 550
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single2 \
#--model tiny_deeplabv3 \
#--checkpoint_i 614
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_multi \
#--model tiny_deeplabv3 \
#--checkpoint_i 554




##f1
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_single1 \
#--model pspnet \
#--checkpoint_i 579
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_single2 \
#--model pspnet \
#--checkpoint_i 466
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_multi \
#--model pspnet \
#--checkpoint_i 564

###f2
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_single2 \
#--model pspnet \
#--checkpoint_i 446
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_multi \
#--model pspnet \
#--checkpoint_i 462




##f1
## DANet 127 169 187          -> 568 610 628
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_single1 \
#--model DANet \
#--checkpoint_i 568
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_single2 \
#--model DANet \
#--checkpoint_i 610
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_multi \
#--model DANet \
#--checkpoint_i 628

###f2
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_multi \
#--model DANet \
#--checkpoint_i 626
#
###f3
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_multi \
#--model DANet \
#--checkpoint_i 576


#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_single1 \
#--model FCN \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_single2 \
#--model FCN \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_multi \
#--model FCN \
#--checkpoint_i 640
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single1 \
#--model Unet \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single2 \
#--model Unet \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_multi \
#--model Unet \
#--checkpoint_i 640
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single1 \
#--model tiny_deeplabv3 \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single2 \
#--model tiny_deeplabv3 \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_multi \
#--model tiny_deeplabv3 \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_single1 \
#--model pspnet \
#--checkpoint_i 640
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_single2 \
#--model pspnet \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_multi \
#--model pspnet \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_single1 \
#--model DANet \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_single2 \
#--model DANet \
#--checkpoint_i 640
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_multi \
#--model DANet \
#--checkpoint_i 640


#f1
# FCN 127 73 101             -> 568 514 (542)
# Unet 173 21 37             -> 614 462 (478)
# tiny 133 161 113             -> 574 602 (554)
# pspnet 199 95 53             -> (640) 536 494
# DANet 175 131 135             -> 616 572 (576)

#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_single1 \
#--model FCN \
#--checkpoint_i 568
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_single2 \
#--model FCN \
#--checkpoint_i 514
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single1 \
#--model Unet \
#--checkpoint_i 614
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single2 \
#--model Unet \
#--checkpoint_i 462
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single1 \
#--model tiny_deeplabv3 \
#--checkpoint_i 574
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single2 \
#--model tiny_deeplabv3 \
#--checkpoint_i 602
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_single2 \
#--model pspnet \
#--checkpoint_i 536
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_multi \
#--model pspnet \
#--checkpoint_i 494
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_single1 \
#--model DANet \
#--checkpoint_i 616
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_single2 \
#--model DANet \
#--checkpoint_i 572
#



##f2
# FCN 173 23 167             -> 614 464 464
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


#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_single1 \
#--model FCN \
#--checkpoint_i 614
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_single2 \
#--model FCN \
#--checkpoint_i 464
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_multi \
#--model FCN \
#--checkpoint_i 608
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single1 \
#--model Unet \
#--checkpoint_i 568
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single2 \
#--model Unet \
#--checkpoint_i 532
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_multi \
#--model Unet \
#--checkpoint_i 576
#
#
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single2 \
#--model tiny_deeplabv3 \
#--checkpoint_i 500
#
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_single1 \
#--model pspnet \
#--checkpoint_i 446
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/pspnet_single2 \
#--model pspnet \
#--checkpoint_i 488
#
#
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_single1 \
#--model DANet \
#--checkpoint_i 610
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_single2 \
#--model DANet \
#--checkpoint_i 516
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_multi \
#--model DANet \
#--checkpoint_i 474
#
###f3
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/Unet_single2 \
#--model Unet \
#--checkpoint_i 457
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/tiny_deeplabv3_single1 \
#--model tiny_deeplabv3 \
#--checkpoint_i 614
#
#python test_pred_mask.py \
#--crop_height 512 \
#--crop_width 512 \
#--gpu 5 \
#--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/DANet_single2 \
#--model DANet \
#--checkpoint_i 578


python test_pred_mask.py \
--crop_height 512 \
--crop_width 512 \
--gpu 5 \
--ckpt /2_data/yym_workspcae/exp/thesis_hp_3/FCN_multi \
--model FCN \
--checkpoint_i 608
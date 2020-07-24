
# FCN 73 173 185             -> 514 614 626
# Unet 15 159 37             -> 456 600 478
# deeplabv3 177 193 119      -> 618 634 560
# pspnet 138 25 123          -> 579 466 564
# DANet 127 169 187          -> 568 610 628
# add 441



## FCN 73 173 185             -> 514 614 626
#python take_post_groundtruth_loop.py \
#--model_task FCN_single1 \
#--checkpoint_i 514
#
#python take_post_groundtruth_loop.py \
#--model_task FCN_single2 \
#--checkpoint_i 614
#
#python take_post_groundtruth_loop.py \
#--model_task FCN_multi \
#--checkpoint_i 626
#
#
###f2
#python take_post_groundtruth_loop.py \
#--model_task   FCN_single1 \
#--checkpoint_i 538
#
#python take_post_groundtruth_loop.py \
#--model_task   FCN_multi \
#--checkpoint_i 542
#
###f3
#python take_post_groundtruth_loop.py \
#--model_task   FCN_single1 \
#--checkpoint_i 602
#
#
#
### Unet 15 159 37             -> 456 600 478
#python take_post_groundtruth_loop.py \
#--model_task Unet_single1 \
#--checkpoint_i 456
#
#python take_post_groundtruth_loop.py \
#--model_task Unet_single2 \
#--checkpoint_i 600
#
#python take_post_groundtruth_loop.py \
#--model_task Unet_multi \
#--checkpoint_i 478
#
#
###f2
#python take_post_groundtruth_loop.py \
#--model_task   Unet_single2 \
#--checkpoint_i 568
#
#
###f3
#python take_post_groundtruth_loop.py \
#--model_task   Unet_single1 \
#--checkpoint_i 550
#
#
#python take_post_groundtruth_loop.py \
#--model_task   Unet_multi \
#--checkpoint_i 614
#
#
#
## deeplabv3 177 193 119      -> 618 634 560
#python take_post_groundtruth_loop.py \
#--model_task tiny_deeplabv3_single1 \
#--checkpoint_i 618
#
#python take_post_groundtruth_loop.py \
#--model_task tiny_deeplabv3_single2 \
#--checkpoint_i 634
#
#python take_post_groundtruth_loop.py \
#--model_task tiny_deeplabv3_multi \
#--checkpoint_i 560
#
###f2
#python take_post_groundtruth_loop.py \
#--model_task   tiny_deeplabv3_single1 \
#--checkpoint_i 550
#
#python take_post_groundtruth_loop.py \
#--model_task   tiny_deeplabv3_single2 \
#--checkpoint_i 614
#
#python take_post_groundtruth_loop.py \
#--model_task   tiny_deeplabv3_multi \
#--checkpoint_i 554
#
#
#
## pspnet 138 25 123          -> 579 466 564
## add 441
#python take_post_groundtruth_loop.py \
#--model_task pspnet_single1 \
#--checkpoint_i 579
#
#python take_post_groundtruth_loop.py \
#--model_task pspnet_single2 \
#--checkpoint_i 466
#
#python take_post_groundtruth_loop.py \
#--model_task pspnet_multi \
#--checkpoint_i 564
#
###f2
#python take_post_groundtruth_loop.py \
#--model_task   pspnet_single2 \
#--checkpoint_i 446
#
#python take_post_groundtruth_loop.py \
#--model_task   pspnet_multi \
#--checkpoint_i 462
#
#
#
## DANet 127 169 187          -> 568 610 628
#python take_post_groundtruth_loop.py \
#--model_task DANet_single1 \
#--checkpoint_i 568
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_single2 \
#--checkpoint_i 610
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_multi \
#--checkpoint_i 628
#
###f2
#python take_post_groundtruth_loop.py \
#--model_task   DANet_multi \
#--checkpoint_i 626
#
###f3
#python take_post_groundtruth_loop.py \
#--model_task   DANet_multi \
#--checkpoint_i 576


#
#python take_post_groundtruth_loop.py \
#--model_task FCN_single1 \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task FCN_single2 \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task FCN_multi \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task Unet_single1 \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task Unet_single2 \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task Unet_multi \
#--checkpoint_i 640
#
#
#python take_post_groundtruth_loop.py \
#--model_task tiny_deeplabv3_single1 \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task tiny_deeplabv3_single2 \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task tiny_deeplabv3_multi \
#--checkpoint_i 640
#
#
#python take_post_groundtruth_loop.py \
#--model_task pspnet_single1 \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task pspnet_single2 \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task pspnet_multi \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_single1 \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_single2 \
#--checkpoint_i 640
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_multi \
#--checkpoint_i 640


#f1
# FCN 127 73 101             -> 568 514 (542)
# Unet 173 21 37             -> 614 462 (478)
# tiny 133 161 113             -> 574 602 (554)
# pspnet 199 95 53             -> (640) 536 494
# DANet 175 131 135             -> 616 572 (576)

#python take_post_groundtruth_loop.py \
#--model_task FCN_single1 \
#--checkpoint_i 568
#
#python take_post_groundtruth_loop.py \
#--model_task FCN_single2 \
#--checkpoint_i 514
#
#
#python take_post_groundtruth_loop.py \
#--model_task Unet_single1 \
#--checkpoint_i 614
#
#python take_post_groundtruth_loop.py \
#--model_task Unet_single2 \
#--checkpoint_i 462
#
#
#python take_post_groundtruth_loop.py \
#--model_task tiny_deeplabv3_single1 \
#--checkpoint_i 574
#
#python take_post_groundtruth_loop.py \
#--model_task tiny_deeplabv3_single2 \
#--checkpoint_i 602
#
#
#python take_post_groundtruth_loop.py \
#--model_task pspnet_single2 \
#--checkpoint_i 536
#
#python take_post_groundtruth_loop.py \
#--model_task pspnet_multi \
#--checkpoint_i 494
#
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_single1 \
#--checkpoint_i 616
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_single2 \
#--checkpoint_i 572



##f2
# FCN 173 23 167             -> 614 464 608
# Unet 127 91 135             -> 568 532 576
# tiny 133 59 113             -> (574) 500 (554)
# pspnet 5 47 53             -> 446 488 (494)
# DANet 169 75 33             -> 610 516 474

#python take_post_groundtruth_loop.py \
#--model_task FCN_single1 \
#--checkpoint_i 614
#
#python take_post_groundtruth_loop.py \
#--model_task FCN_single2 \
#--checkpoint_i 464
#
#python take_post_groundtruth_loop.py \
#--model_task FCN_multi \
#--checkpoint_i 608
#
#python take_post_groundtruth_loop.py \
#--model_task Unet_single1 \
#--checkpoint_i 568
#
#python take_post_groundtruth_loop.py \
#--model_task Unet_single2 \
#--checkpoint_i 532
#
#python take_post_groundtruth_loop.py \
#--model_task Unet_multi \
#--checkpoint_i 576
#
#
#python take_post_groundtruth_loop.py \
#--model_task tiny_deeplabv3_single2 \
#--checkpoint_i 500
#
#
#python take_post_groundtruth_loop.py \
#--model_task pspnet_single1 \
#--checkpoint_i 446
#
#python take_post_groundtruth_loop.py \
#--model_task pspnet_single2 \
#--checkpoint_i 488
#
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_single1 \
#--checkpoint_i 610
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_single2 \
#--checkpoint_i 516
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_multi \
#--checkpoint_i 474
#
#
#
## ##f3
## FCN 173 23 167             -> (614 464 608)
## Unet 127 16 135             -> (568) 457 (576)
## tiny 173 59 113             -> 614 (500 554)
## pspnet 5 47 53             -> (446 488 494)
## DANet 169 137 33             -> (610) 578 (474)
#python take_post_groundtruth_loop.py \
#--model_task Unet_single2 \
#--checkpoint_i 457
#
#python take_post_groundtruth_loop.py \
#--model_task tiny_deeplabv3_single1 \
#--checkpoint_i 614
#
#python take_post_groundtruth_loop.py \
#--model_task DANet_single2 \
#--checkpoint_i 578



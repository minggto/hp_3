
# FCN 73 173 185             -> 514 614 626
# Unet 15 159 37             -> 456 600 478
# deeplabv3 177 193 119      -> 618 634 560
# pspnet 138 25 123          -> 579 466 564
# add 441




python thesis_hp3_show.py \
--model_task   FCN_single1 \
--checkpoint_i 514

python thesis_hp3_show.py \
--model_task   FCN_single1 \
--checkpoint_i 614

#python thesis_hp3_show.py \
#--model_task   FCN_multi \
#--checkpoint_i 626

#python thesis_hp3_show.py \
#--model_task   Unet_multi \
#--checkpoint_i 478
#
#python thesis_hp3_show.py \
#--model_task   tiny_deeplabv3_multi \
#--checkpoint_i 560

#python thesis_hp3_show.py \
#--model_task   pspnet_multi \
#--checkpoint_i 564
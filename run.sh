
interval=4
model_name="MedSAM_bbox"
nums=32
prompt="bbox"
alpha_percent=1
dice_percent=0

python main.py  --directory  "Polyp"             --dice_percent ${dice_percent}    --parts "CVC-ClinicDB" "CVC-ColonDB" "ETIS-LaribPolypDB" "Kvasir"  "CVC-300"   --interval ${interval}     --model_name   ${model_name}    --nums ${nums}   --prompt ${prompt}    --alpha_percent ${alpha_percent}
python main.py  --directory  "BUSI"              --dice_percent ${dice_percent}     --parts "benign"                                                               --interval ${interval}      --model_name   ${model_name}     --nums ${nums}   --prompt ${prompt}    --alpha_percent ${alpha_percent}
python main.py  --directory  "BUSI"              --dice_percent ${dice_percent}     --parts "malignant"                                                            --interval ${interval}      --model_name   ${model_name}      --nums ${nums}   --prompt ${prompt}   --alpha_percent ${alpha_percent}
python main.py  --directory  "GlaS"              --dice_percent ${dice_percent}    --parts "benign"                                                               --interval ${interval}      --model_name  ${model_name}      --nums ${nums}   --prompt ${prompt}    --alpha_percent ${alpha_percent}
python main.py  --directory  "GlaS"              --dice_percent ${dice_percent}     --parts "malignant"                                                            --interval ${interval}      --model_name  ${model_name}     --nums ${nums}   --prompt ${prompt}    --alpha_percent ${alpha_percent}
python main.py  --directory  "fluidchallenge"    --dice_percent ${dice_percent}     --parts "cirrus"                                                               --interval ${interval}      --model_name   ${model_name}    --nums ${nums}   --prompt ${prompt}   --alpha_percent ${alpha_percent}
python main.py  --directory  "fluidchallenge"    --dice_percent ${dice_percent}     --parts "topcon"                                                               --interval ${interval}      --model_name  ${model_name}     --nums ${nums}   --prompt ${prompt}    --alpha_percent ${alpha_percent}
python main.py  --directory  "fluidchallenge"    --dice_percent ${dice_percent}     --parts "spectralis"                                                           --interval ${interval}     --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}    --alpha_percent ${alpha_percent}


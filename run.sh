interval=1
dice_percent=1
nums=32
noise=0
model_name="SAM"
prompt="bbox"
four_channel=1
noise_type="both"
noise_ratio=1

python main.py  --directory  "Polyp"          --parts  "CVC-ClinicDB" "CVC-ColonDB" "ETIS-LaribPolypDB" "Kvasir"  "CVC-300"      --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}   --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "BUSI"           --parts "benign"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "BUSI"           --parts "malignant"                                                               --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "GlaS"           --parts "benign"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "GlaS"           --parts "malignant"                                                               --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "cirrus"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "topcon"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "spectralis"                                                              --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}

interval=1
dice_percent=1
nums=32
noise=0
model_name="SAM"
prompt="point"
four_channel=0
noise_type="both"
noise_ratio=1

python main.py  --directory  "Polyp"          --parts  "CVC-ClinicDB" "CVC-ColonDB" "ETIS-LaribPolypDB" "Kvasir"  "CVC-300"     --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "BUSI"           --parts "benign"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "BUSI"           --parts "malignant"                                                               --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "GlaS"           --parts "benign"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "GlaS"           --parts "malignant"                                                               --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "cirrus"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "topcon"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "spectralis"                                                              --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}


interval=1
dice_percent=1
nums=32
noise=0
model_name="MedSAM_bbox"
prompt="bbox"
four_channel=1
noise_type="both"
noise_ratio=1


python main.py  --directory  "Polyp"          --parts  "CVC-ClinicDB" "CVC-ColonDB" "ETIS-LaribPolypDB" "Kvasir"  "CVC-300"     --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "BUSI"           --parts "benign"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "BUSI"           --parts "malignant"                                                               --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "GlaS"           --parts "benign"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "GlaS"           --parts "malignant"                                                               --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "cirrus"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "topcon"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "spectralis"                                                              --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear  --noise_type ${noise_type}    --noise ${noise}  --noise_ratio ${noise_ratio}



interval=1
dice_percent=1
nums=32
noise=0
model_name="MedSAM_point"
prompt="point"
four_channel=0
noise_type="both"
noise_ratio=1


python main.py  --directory  "Polyp"          --parts  "CVC-ClinicDB" "CVC-ColonDB" "ETIS-LaribPolypDB" "Kvasir"  "CVC-300"     --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear    --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "BUSI"           --parts "benign"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear    --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "BUSI"           --parts "malignant"                                                               --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear    --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "GlaS"           --parts "benign"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear    --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "GlaS"           --parts "malignant"                                                               --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear    --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "cirrus"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear    --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "topcon"                                                                  --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear    --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}
python main.py  --directory  "fluidchallenge" --parts "spectralis"                                                              --four_channel ${four_channel}  --dice_percent ${dice_percent}   --interval ${interval}   --model_name   ${model_name}   --nums ${nums}   --prompt ${prompt}   --bilinear    --noise_type ${noise_type}    --noise ${noise} --noise_ratio ${noise_ratio}



# directory = "BUSI"
# "benign","malignant"
# directory = "GlaS"
# "benign", "malignant"
# directory="Polyp"
# "CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB", "Kvasir", "CVC-300"
# directory="fluidchallenge"
# "cirrus","topcon","spectralis"

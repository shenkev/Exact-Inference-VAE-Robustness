for i in $(seq 0 199)
do
   export log_name='log.txt'
   export image_index=$i
   python hacked_hmc_jernej_batchsize_one.py
done

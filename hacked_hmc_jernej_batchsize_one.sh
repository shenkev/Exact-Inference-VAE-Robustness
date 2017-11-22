for i in $(seq 0 991)
do
   export image_index=$i
   python hacked_hmc_jernej_batchsize_one.py
done
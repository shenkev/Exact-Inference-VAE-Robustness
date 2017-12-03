start=`date +%s`
for i in $(seq 600 799)
do
   export log_name='log600.txt'
   export image_index=$i
   python hacked_hmc_jernej_batchsize_one.py
done
end=`date +%s`
runtime=$((end-start))
echo 'Running time: ' $runtime

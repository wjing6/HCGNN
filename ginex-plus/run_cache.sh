start_gpu=0
for i in 0.005 0.02 0.04 0.06 0.08;
do
    log_file='log_'$i'.txt'
	python3 test_sample.py --prop=$i --gpu=$start_gpu > $log_file&
    echo "dispatch $i"
    start_gpu+=1
done
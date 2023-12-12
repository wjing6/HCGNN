for i in 1024 512 256;
do
	echo $i threads;
	python3 ogbn_products_sage_ginex.py --verbose --train-only --ginex-num-threads=$i
done

# 设置实验结果输出文件
output_file="exp_result.txt"

# 清空输出文件，如果文件已存在
> "$output_file"

# 定义一组实验命令或函数
run_experiment() {
    echo "Running Experiment..."
    python sage_ginex_gpu_sample.py --verbose --embedding-sizes=$1,$2 --gpu=0 --stale-thre=$3 --dataset=$4>> "$output_file"
}

run_experiment 0.001 0.005 10 ogbn-papers100M
run_experiment 0.001 0.01 10 ogbn-papers100M
run_experiment 0.005 0.05 10 ogbn-papers100M
run_experiment 0.001 0.005 20 ogbn-papers100M
run_experiment 0.001 0.01 20 ogbn-papers100M
run_experiment 0.005 0.05 20 ogbn-papers100M
run_experiment 0.001 0.005 50 ogbn-papers100M
run_experiment 0.001 0.01 50 ogbn-papers100M
run_experiment 0.005 0.05 50 ogbn-papers100M

echo "All experiments completed. Results saved to $output_file."
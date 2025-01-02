# 设置实验结果输出文件
output_file="exp_result.txt"

# 清空输出文件，如果文件已存在
> "$output_file"

# 定义一组实验命令或函数
run_experiment() {
    local layer_1_emb=$1    
    local layer_2_emb=$2
    local stale_thre=$3
    echo "Running Experiment..."
    python sage_ginex_gpu_sample.py --verbose --embedding-sizes=$1,$2 --gpu=0 --stale-thre=$3 >> "$output_file"
}

run_experiment 0.001 0.005 10
run_experiment 0.001 0.01 10
run_experiment 0.005 0.01 10
run_experiment 0.005 0.05 10
run_experiment 0.001 0.005 20
run_experiment 0.001 0.01 20
run_experiment 0.005 0.01 20
run_experiment 0.005 0.05 20
run_experiment 0.001 0.005 50
run_experiment 0.001 0.01 50
run_experiment 0.005 0.01 50
run_experiment 0.005 0.05 50

echo "All experiments completed. Results saved to $output_file."
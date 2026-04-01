#!/bin/bash

#SBATCH --job-name=eval_pi0_base           # 作业名
#SBATCH --partition=gpuB                # 使用 gpuB 分区
#SBATCH --nodes=1                       # 只使用一个节点
#SBATCH --ntasks-per-node=1             # 一个训练任务（DDP rank 0）
#SBATCH --cpus-per-task=12              # 根据卡配置调整，一般 12 核对一张 A100 是合理的
#SBATCH --gres=gpu:1                    # 请求 1 张 GPU
#SBATCH -o %j.out                       # 标准输出
#SBATCH -e %j.err                       # 错误输出
#SBATCH --mail-type=BEGIN,END,FAIL      # 可以修改，任务开始，结束，失败时邮件通知
#SBATCH --mail-user=xxxx@seu.edu.cn     # 可以修改，邮件通知邮箱      

# 加载环境模块
module load anaconda3


# 初始化 Conda
__conda_setup="$('/seu_share/apps/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/seu_share/apps/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/seu_share/apps/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/seu_share/apps/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# 激活环境
conda activate openpi
cd /seu_share2/home/linli/213221090/RoboCerebra/evaluation

export HF_HOME=/seu_share2/home/linli/213221090/hf_cache
export HF_HUB_OFFLINE=1  
export TRANSFORMERS_OFFLINE=1  
export MUJOCO_GL=egl
export PYTHONPATH=/seu_share2/home/linli/213221090/RoboCerebra/LIBERO:$PYTHONPATH 
export PYTHONPATH=/seu_share2/home/linli/213221090/openpi/src:$PYTHONPATH

python eval_pi0.py --use_task_tree_desc_baseline True --pretrained_checkpoint /seu_share2/home/linli/213221090/.cache/openpi/openpi-assets/checkpoints/pi0_libero --openpi_config_name pi0_libero --robocerebra_root /seu_share2/home/linli/213221090/RoboCerebra/RoboCerebra/RoboCerebraBench --init_files_root /seu_share2/home/linli/213221090/RoboCerebra/RoboCerebra/RoboCerebraBench/init_files --num_trials_per_task 1 --task_types ["Ideal"]


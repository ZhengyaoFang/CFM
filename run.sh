
export HUGGINGFACE_HUB_TOKEN=YOUR_TOKEN
export WANDB_API_KEY=YOUR_TOKEN

export WANDB_PROJECT="CFM"
export WANDB_RUN_NAME="CFM_7B"

PYTHONPATH=. deepspeed cfm/train.py --config cfm/config/CFM_7B.yaml
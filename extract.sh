# 默认就取第 700 个 epoch（有就用；没有就回退到 final 日志）
python3 collect_hypertune_epoch.py

# # 强制必须存在第 700 个 epoch（否则该 run 跳过；不回退）
# python3 collect_hypertune_epoch.py --strict-epoch

# # 改变要取的 epoch
# python3 collect_hypertune_epoch.py --epoch 500

# 画 loss 热力图（也支持 l2_loss）
python3 collect_hypertune_epoch.py --metric final_loss

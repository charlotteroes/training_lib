set -x

rm *.bin
python mla_rope_train.py

python ./fp_diff.py refq.bin ../forward/refq.bin
python ./fp_diff.py refkv.bin ../forward/refkv.bin

python ./fp_diff.py q_grad_ref.bin ../backward/q_grad_ref.bin
python ./fp_diff.py kv_grad_ref.bin ../backward/kv_grad_ref.bin
python ./fp_diff.py k_pe_grad_ref.bin ../backward/k_pe_grad_ref.bin

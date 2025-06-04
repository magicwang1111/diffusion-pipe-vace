from safetensors.torch import load_file

ckpt_path = "/mnt/data/shared_data/ComfyuiModels/diffusion_models/wan/wan2.1_i2v_720p_14B_bf16.safetensors"
state_dict = load_file(ckpt_path, device="cpu")

with open("wan_i2v_keys.txt", "w", encoding="utf-8") as f:
    for k in state_dict.keys():
        f.write(k + "\n")
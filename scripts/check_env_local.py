import json
import subprocess

info = {}

try:
    import bitsandbytes as bnb
    info['bitsandbytes'] = getattr(bnb, '__version__', 'unknown')
except Exception as e:
    info['bitsandbytes_error'] = str(e)

try:
    cmd = ['nvidia-smi', '-L']
    r = subprocess.run(cmd, capture_output=True, text=True)
    info['nvidia_smi_L'] = r.stdout.strip()
except Exception as e:
    info['nvidia_smi_L_error'] = str(e)

try:
    cmd = ['nvidia-smi']
    r = subprocess.run(cmd, capture_output=True, text=True)
    info['nvidia_smi_full'] = r.stdout.strip()
except Exception as e:
    info['nvidia_smi_full_error'] = str(e)

try:
    cmd = ['nvcc', '--version']
    r = subprocess.run(cmd, capture_output=True, text=True)
    info['nvcc'] = r.stdout.strip()
except Exception as e:
    info['nvcc_error'] = str(e)

print(json.dumps(info, ensure_ascii=False))

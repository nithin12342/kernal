import os, subprocess

print("=== NVIDIA GPU SILICON CHECK ===")
nvidia_smi_bin = "/usr/bin/nvidia-smi" if os.path.exists("/usr/bin/nvidia-smi") else "nvidia-smi"
res_gpu = subprocess.run([nvidia_smi_bin], capture_output=True, text=True)
print(res_gpu.stdout if res_gpu.stdout else res_gpu.stderr)

print("=== RUST TOOLCHAIN SETUP ===")
p = subprocess.run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y", shell=True, capture_output=True, text=True)
print("Rustup output:\n", p.stdout[-250:] if p.stdout else p.stderr)

res = subprocess.run(["/root/.cargo/bin/rustc", "--version"], capture_output=True, text=True)
print("Rustc version:", res.stdout.strip())

res_cargo = subprocess.run(["/root/.cargo/bin/cargo", "--version"], capture_output=True, text=True)
print("Cargo version:", res_cargo.stdout.strip())

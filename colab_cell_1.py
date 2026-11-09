import subprocess, sys

print("=== NVIDIA GPU SILICON CHECK ===")
res_gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(res_gpu.stdout)
assert res_gpu.returncode == 0, "No GPU detected!"

print("=== RUST TOOLCHAIN SETUP ===")
rustc_check = subprocess.run(["/root/.cargo/bin/rustc", "--version"], capture_output=True, text=True)
if rustc_check.returncode != 0:
    print("Installing Rust toolchain...")
    p = subprocess.run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y", shell=True, capture_output=True, text=True)
    print(p.stdout[-300:] if p.stdout else p.stderr)
else:
    print("Rust already installed:", rustc_check.stdout.strip())

res_v = subprocess.run(["/root/.cargo/bin/rustc", "--version"], capture_output=True, text=True)
print("Active Rustc:", res_v.stdout.strip())

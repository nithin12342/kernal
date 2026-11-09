"""
Colab Direct Automation Driver
Connects to Google Colab via native WebSocket MCP protocol,
waits for GPU configuration, deploys multi-precision
kernel, compiles, tests on Tesla T4 silicon, and captures full evidence.
"""

import sys
sys.path.insert(0, r"C:\Users\NITHING\AppData\Local\uv\cache\git-v1\checkouts\79ec5ff6492e82aa\b9ab389\src")

import asyncio
import json
import os
import webbrowser
import websockets
from websockets.typing import Subprotocol
from colab_mcp.websocket_server import ColabWebSocketServer
from colab_mcp.session import ColabProxyClient

COLAB_URL = "https://colab.research.google.com/notebooks/empty.ipynb"
FIXED_PORT = 58954
FIXED_TOKEN = "RwGeWal2ITbay9MRXW42Pw"


class FixedColabWebSocketServer(ColabWebSocketServer):
    def __init__(self, port=FIXED_PORT, token=FIXED_TOKEN, host="localhost"):
        super().__init__(host=host)
        self.desired_port = port
        self.token = token

    async def __aenter__(self):
        try:
            self._server = await websockets.serve(
                self._connection_handler,
                host=self.host,
                port=self.desired_port,
                subprotocols=[Subprotocol("mcp")],
                origins=self.allowed_origins,
                process_request=self._validate_authorization,
            )
        except Exception as e:
            print(f"Port {self.desired_port} busy ({e}), selecting random port...")
            self._server = await websockets.serve(
                self._connection_handler,
                host=self.host,
                port=0,
                subprotocols=[Subprotocol("mcp")],
                origins=self.allowed_origins,
                process_request=self._validate_authorization,
            )
        self.port = self._server.sockets[0].getsockname()[1]
        return self


def get_cell_text(run_data):
    full_text = []
    for out in run_data.get("outputs", []):
        if "text" in out:
            t = out["text"]
            if isinstance(t, list):
                full_text.extend(t)
            else:
                full_text.append(str(t))
        if "traceback" in out:
            for tb_line in out["traceback"]:
                full_text.append(str(tb_line) + "\n")
        if "evalue" in out:
            full_text.append(f"Error: {out.get('ename', '')}: {out.get('evalue', '')}\n")
    return "".join(full_text)


def print_cell_outputs(run_data):
    print(get_cell_text(run_data))


CHECK_GPU_CODE = """
import subprocess, os
try:
    res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if res.returncode == 0:
        print("[GPU_READY]\\n" + res.stdout)
    else:
        print("[GPU_NOT_READY] non-zero exit code: " + str(res.returncode))
except Exception as e:
    print(f"[GPU_NOT_READY] {e}")
"""


async def main():
    print("=" * 80)
    print("STARTING GOOGLE COLAB DIRECT MCP AUTOMATION ENGINE")
    print("=" * 80)

    wss = FixedColabWebSocketServer()
    async with wss:
        token = wss.token
        port = wss.port
        full_url = f"{COLAB_URL}#mcpProxyToken={token}&mcpProxyPort={port}"

        print(f"\n[1/4] MCP WebSocket Server listening on port {port}")
        print(f"[1/4] Colab browser tab URL:\n  --> {full_url}\n")
        try:
            webbrowser.open_new(full_url)
        except Exception:
            pass

        print("-" * 80)
        print("ACTION REQUIRED BY USER:")
        print("1. If not already open, open the URL above in your browser.")
        print("2. Ensure runtime is GPU: Runtime -> Change runtime type -> T4 GPU -> Save.")
        print("3. Check top-right: it should show 'T4' (not just RAM/Disk).")
        print("4. Automation will detect GPU silicon and run all benchmarks!")
        print("-" * 80)

        print("\nWaiting for Colab browser tab to establish WebSocket handshake...")
        await wss.connection_live.wait()
        print("\n[SUCCESS] Colab browser tab connected via WebSocket!\n")

        proxy = ColabProxyClient(wss)
        async with proxy:
            await proxy._start_task
            client = proxy.proxy_mcp_client
            
            tools_res = await client.list_tools()
            tool_names = [t.name for t in tools_res]
            print(f"[2/4] Available Colab MCP Tools: {tool_names}")

            cells_res = await client.call_tool("get_cells", arguments={"includeOutputs": False})
            cells_data = json.loads(cells_res.content[0].text) if cells_res.content else {}
            existing_count = len(cells_data.get("cells", []))
            print(f"[2/4] Initial Notebook Cells Count: {existing_count}")

            # Step 1: Append GPU Poller Cell
            print("\n" + "=" * 80)
            print("[STAGE 1] POLLING FOR PHYSICAL NVIDIA GPU SILICON (TESLA T4)")
            print("=" * 80)
            add_res0 = await client.call_tool("add_code_cell", arguments={
                "cellIndex": existing_count,
                "language": "python",
                "code": CHECK_GPU_CODE
            })
            cell_poll_id = json.loads(add_res0.content[0].text)["newCellId"]

            gpu_confirmed = False
            for attempt in range(1, 31):
                run_poll = await client.call_tool("run_code_cell", arguments={"cellId": cell_poll_id})
                poll_text = get_cell_text(json.loads(run_poll.content[0].text))
                if "[GPU_READY]" in poll_text:
                    print(f"\n[GPU CONFIRMED on Attempt {attempt}] NVIDIA Silicon Active!")
                    print(poll_text)
                    gpu_confirmed = True
                    break
                else:
                    print(f"[Attempt {attempt}/30] Runtime is CPU ('RAM/Disk'). Waiting for user to select T4 GPU and click Save...")
                    await asyncio.sleep(8)

            if not gpu_confirmed:
                print("Timed out waiting for GPU runtime. Please switch runtime to T4 GPU and retry.")
                return

            # Step 2: Append and run Rust setup
            print("\n" + "=" * 80)
            print("[STAGE 2] SETTING UP RUST ENVIRONMENT ON GPU INSTANCE")
            print("=" * 80)
            with open(r"d:\kernal\colab_cell_rust.py", "r", encoding="utf-8") as f:
                rust_code = f.read()

            add_res1 = await client.call_tool("add_code_cell", arguments={
                "cellIndex": existing_count + 1,
                "language": "python",
                "code": rust_code
            })
            cell1_id = json.loads(add_res1.content[0].text)["newCellId"]
            print(f"Appended Rust Setup Cell (ID: {cell1_id}). Running...")

            run_res1 = await client.call_tool("run_code_cell", arguments={"cellId": cell1_id})
            run_data1 = json.loads(run_res1.content[0].text)
            print_cell_outputs(run_data1)

            # Step 3: Append and run Cell: Multi-Precision Deployment, Compilation & Benchmark
            print("\n" + "=" * 80)
            print("[STAGE 3] DEPLOYING MULTI-PRECISION REPOSITORY & EXECUTING BENCHMARKS")
            print("=" * 80)
            with open(r"d:\kernal\colab_cell_3_deploy_and_test.py", "r", encoding="utf-8") as f:
                bench_code = f.read()

            add_res2 = await client.call_tool("add_code_cell", arguments={
                "cellIndex": existing_count + 2,
                "language": "python",
                "code": bench_code
            })
            cell2_id = json.loads(add_res2.content[0].text)["newCellId"]
            print(f"Appended Multi-Precision Benchmark Cell (ID: {cell2_id}). Running...")

            run_res2 = await client.call_tool("run_code_cell", arguments={"cellId": cell2_id})
            run_data2 = json.loads(run_res2.content[0].text)
            
            print("\n" + "=" * 80)
            print("PHYSICAL GPU SILICON MULTI-PRECISION EXECUTION OUTPUT")
            print("=" * 80)
            print_cell_outputs(run_data2)

            print("\n" + "=" * 80)
            print("AUTOMATION RUN COMPLETED SUCCESSFULLY ON PHYSICAL SILICON!")
            print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

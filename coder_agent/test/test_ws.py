import asyncio
import json
import re
import base64
import websockets
import urllib.parse
import httpx

async def run_shell_ws():
    mcp_url = "http://localhost:8081/mcp"
    uri = "ws://localhost:8081/v1/shell/ws"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as ws:
            print("Connected!")
            
            cwd = "/home/gem/workspace"
            user_command = "for i in {1..5}; do echo 'Streaming: ' $i; sleep 1; done"
            
            # Write stdout/stderr to /tmp/cmd_output.txt while streaming to stdout
            wrapped_user_command = f"( {user_command} ) 2>&1 | tee /tmp/cmd_output.txt"
            
            full_command = f"cd {cwd} && {wrapped_user_command}" if cwd else wrapped_user_command
            encoded_cmd = base64.b64encode(full_command.encode("utf-8")).decode("utf-8")
            
            sentinel_prefix = "___CMD_FINISHED___:"
            wrapper_cmd = f"echo -n '{encoded_cmd}' | base64 -d | bash; echo \"{sentinel_prefix}$?\"\n"
            
            accumulated_output = ""
            exit_code = None
            
            async for msg in ws:
                obj = json.loads(msg)
                if obj.get("type") == "ready":
                    await ws.send(json.dumps({"type": "input", "data": wrapper_cmd}))
                elif obj.get("type") == "output":
                    data = obj.get("data", "")
                    # Stream raw data to stdout in real-time
                    print(data, end="", flush=True)
                    accumulated_output += data
                    
                    match = re.search(r"___CMD_FINISHED___:(\d+)", accumulated_output)
                    if match:
                        exit_code = int(match.group(1))
                        break
            
            print(f"\nCommand finished with exit code: {exit_code}")
            
            # Read clean output from /tmp/cmd_output.txt using HTTP MCP tool
            print("Reading clean output file from sandbox...")
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    mcp_url,
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "sandbox_file_operations",
                            "arguments": {"action": "read", "path": "/tmp/cmd_output.txt"}
                        },
                        "id": 2
                    },
                    headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
                )
                res_obj = resp.json()
                text = res_obj.get("result", {}).get("content", [])[0].get("text", "")
                inner_res = json.loads(text)
                clean_output = inner_res.get("content", "")
                
            print("\n" + "="*40)
            print("CLEAN EXTRACTED OUTPUT:")
            print(repr(clean_output))
            print("="*40)
            
    except Exception as e:
        print(f"\nWS Error: {e}")

asyncio.run(run_shell_ws())

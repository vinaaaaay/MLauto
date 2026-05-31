import json
import base64
import re
import httpx
import websockets
import logging
import urllib.parse
from typing import Tuple

logger = logging.getLogger(__name__)

class LocalSandboxClient:
    """
    Local hybrid adapter: HTTP for files, WebSockets for execution streaming.
    Uses /v1/file/* for file operations and ws://.../v1/shell/ws for shell execution.
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        orig_url = base_url
        scheme = "http"
        ws_scheme = "ws"
        if "://" in base_url:
            parsed = urllib.parse.urlparse(base_url)
            base_url = parsed.netloc
            if parsed.scheme in ["https", "wss"]:
                scheme = "https"
                ws_scheme = "wss"
        
        self.http_url = f"{scheme}://{base_url}"
        self.ws_url = f"{ws_scheme}://{base_url}/v1/shell/ws"
        self.timeout = httpx.Timeout(60.0)
        logger.info(f"Initialized LocalSandboxClient with http_url={self.http_url}, ws_url={self.ws_url}")

    async def read_file(self, path: str) -> str:
        url = f"{self.http_url}/v1/file/read"
        payload = {"file": path}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            res = response.json()
            
            if not res.get("success"):
                raise IOError(f"Failed to read file {path}: {res.get('message')}")
            
            return res.get("data", {}).get("content", "")

    async def write_file(self, path: str, content: str) -> bool:
        import os
        parent_dir = os.path.dirname(path)
        if parent_dir:
            await self.exec_shell(f"mkdir -p {parent_dir}", cwd="")
        await self.exec_shell(f"rm -f {path}", cwd="")

        url = f"{self.http_url}/v1/file/write"
        chunk_size = 50000
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(content), chunk_size):
                chunk = content[i : i + chunk_size]
                payload = {
                    "file": path,
                    "content": chunk,
                    "append": True
                }
                response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                response.raise_for_status()
                res = response.json()
                if not res.get("success"):
                    logger.error(f"Failed to write file chunk to {path}: {res.get('message')}")
                    return False
        return True

    async def exec_shell(self, command: str, cwd: str = "/home/gem/workspace") -> Tuple[bool, str, str]:
        logger.info(f"\\n--- WS Execution: {command} (cwd={cwd}) ---\\n")
        
        wrapped_command = f"( {command} ) 2>&1"
        full_command = f"cd {cwd} && {wrapped_command}" if cwd else wrapped_command
        
        encoded_cmd = base64.b64encode(full_command.encode("utf-8")).decode("utf-8")
        sentinel = "___CMD_FINISHED___:"
        wrapper_cmd = f"echo -n '{encoded_cmd}' | base64 -d | bash; echo \"{sentinel}$?\"\n"

        accumulated_output = ""
        exit_code = 1 

        try:
            async with websockets.connect(self.ws_url) as ws:
                async for msg in ws:
                    obj = json.loads(msg)
                    msg_type = obj.get("type")
                    
                    if msg_type == "ready":
                        await ws.send(json.dumps({"type": "input", "data": wrapper_cmd}))
                        
                    elif msg_type == "ping":
                        await ws.send(json.dumps({"type": "pong", "data": obj.get("data")}))
                        
                    elif msg_type == "output":
                        data = obj.get("data", "")
                        print(data, end="", flush=True)
                        accumulated_output += data
                        
                        clean_output = re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', accumulated_output)
                        match = re.search(r"___CMD_FINISHED___:([0-9]+)", clean_output)
                        if match:
                            exit_code = int(match.group(1))
                            break
                            
        except Exception as e:
            logger.error(f"WS Execution Error: {e}")
            return False, accumulated_output, str(e)
            
        logger.info(f"\\n--- WS Execution Complete (Exit: {exit_code}) ---\\n")
        return (exit_code == 0), accumulated_output, ""

    def _run_sync(self, coro):
        import asyncio
        import threading
        
        result = []
        exception = []
        
        def run_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result.append(loop.run_until_complete(coro))
                loop.close()
            except Exception as e:
                exception.append(e)
                
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception:
            raise exception[0]
        return result[0]

    def read_file_sync(self, path: str) -> str:
        return self._run_sync(self.read_file(path))

    def write_file_sync(self, path: str, content: str) -> bool:
        return self._run_sync(self.write_file(path, content))

    def exec_shell_sync(self, command: str, cwd: str = "/home/gem/workspace") -> Tuple[bool, str, str]:
        return self._run_sync(self.exec_shell(command, cwd))

import asyncio
import sys
import logging
from pathlib import Path

# Ensure FAME root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.coder_agent.tools.ws_sandbox import AgentInfraWSSandbox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_ws_sandbox")

async def test_write_and_read(sandbox: AgentInfraWSSandbox):
    logger.info("Running test_write_and_read...")
    test_path = "/home/gem/workspace/test_file.txt"
    test_content = "Hello from FAME WebSocket client!\nThis is line 2."
    
    # Write file
    success = await sandbox.write_file(test_path, test_content)
    assert success is True, "Write file failed"
    
    # Read file back
    read_content = await sandbox.read_file(test_path)
    assert read_content == test_content, f"Content mismatch: expected {repr(test_content)}, got {repr(read_content)}"
    logger.info("test_write_and_read PASSED!")

async def test_exec_shell_success(sandbox: AgentInfraWSSandbox):
    logger.info("Running test_exec_shell_success...")
    success, stdout, stderr = await sandbox.exec_shell("echo 'Hello World'", cwd="/home/gem/workspace")
    assert success is True, "Command failed"
    assert "Hello World" in stdout, f"Expected 'Hello World' in stdout, got {repr(stdout)}"
    assert stderr == "", f"Expected empty stderr, got {repr(stderr)}"
    logger.info("test_exec_shell_success PASSED!")

async def test_exec_shell_streaming(sandbox: AgentInfraWSSandbox):
    logger.info("Running test_exec_shell_streaming...")
    # This should print numbers 1 to 5 to the console in real-time
    command = "for i in {1..5}; do echo 'Streaming: '$i; sleep 0.2; done"
    success, stdout, stderr = await sandbox.exec_shell(command, cwd="/home/gem/workspace")
    assert success is True, "Streaming command failed"
    for i in range(1, 6):
        assert f"Streaming: {i}" in stdout, f"Expected 'Streaming: {i}' in stdout"
    logger.info("test_exec_shell_streaming PASSED!")

async def test_exec_shell_failure(sandbox: AgentInfraWSSandbox):
    logger.info("Running test_exec_shell_failure...")
    success, stdout, stderr = await sandbox.exec_shell("exit 42", cwd="/home/gem/workspace")
    assert success is False, "Command was expected to fail"
    logger.info("test_exec_shell_failure PASSED!")

async def test_exec_shell_multiline(sandbox: AgentInfraWSSandbox):
    logger.info("Running test_exec_shell_multiline...")
    import textwrap
    multiline_cmd = textwrap.dedent("""\
        cat << 'EOF' > /tmp/multiline_test.sh
        echo "Line 1"
        echo "Line 2"
        EOF
        bash /tmp/multiline_test.sh
    """)
    success, stdout, stderr = await sandbox.exec_shell(multiline_cmd, cwd="/home/gem/workspace")
    assert success is True, "Multiline command failed"
    assert "Line 1" in stdout
    assert "Line 2" in stdout
    logger.info("test_exec_shell_multiline PASSED!")

async def main():
    # Use sandbox port 8081 as per test_ws.py and approved plan
    sandbox = AgentInfraWSSandbox(base_url="localhost:8081")
    
    try:
        await test_write_and_read(sandbox)
        await test_exec_shell_success(sandbox)
        await test_exec_shell_streaming(sandbox)
        await test_exec_shell_failure(sandbox)
        await test_exec_shell_multiline(sandbox)
        logger.info("ALL TESTS PASSED!")
    except AssertionError as e:
        logger.error(f"TEST FAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

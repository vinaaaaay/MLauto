import unittest
import uuid
import sys
from pathlib import Path

# Ensure parent directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.coder_agent.tools import SyncMCPClient

class TestLiveSandbox(unittest.TestCase):
    def setUp(self):
        # Connect to the live Docker container (running on port 8081)
        self.client = SyncMCPClient("http://localhost:8081/mcp")
        self.client.connect()
        
        # Create a unique directory for each test run to avoid collisions
        self.test_dir = f"/home/gem/workspace/test_env_{uuid.uuid4().hex[:6]}"
        self.client.exec_shell(f"mkdir -p {self.test_dir}")

    def tearDown(self):
        # Clean up the test directory after each test
        self.client.exec_shell(f"rm -rf {self.test_dir}")
        self.client.disconnect()

    def test_live_file_persistence(self):
        """Verifies the agent can write and accurately read back files."""
        test_file = f"{self.test_dir}/hello.txt"
        test_content = "Sanity check: Agent data injection successful."
        
        # Action: Write
        self.client.write_file(test_file, test_content)
        
        # Assert: Read back and match
        read_content = self.client.read_file(test_file)
        self.assertEqual(read_content, test_content)

    def test_execution_context(self):
        """Verifies the pwd and whoami execution defaults and user permissions."""
        # Action: Run pwd and whoami
        success, output, error = self.client.exec_shell("pwd")
        self.assertTrue(success, f"pwd execution failed: {error}")
        self.assertEqual(output.strip(), "/home/gem/workspace")

        success, output, error = self.client.exec_shell("whoami")
        self.assertTrue(success, f"whoami execution failed: {error}")
        self.assertEqual(output.strip(), "gem")

    def test_live_execution_state(self):
        """Verifies the shell can see and execute files the API just wrote."""
        script_path = f"{self.test_dir}/calculate.py"
        script_content = "print(10 * 5)"
        
        # Action: Write script
        self.client.write_file(script_path, script_content)
        
        # Action: Execute script via bash
        success, output, error = self.client.exec_shell(f"python3 {script_path}")
        
        # Assert
        self.assertTrue(success, f"Execution failed with error: {error}")
        self.assertEqual(output.strip(), "50")

    def test_error_handling(self):
        """Verifies that non-existent files or malformed commands return structured JSON failures rather than throwing Python exceptions."""
        # Action 1: Read a non-existent file directly via call_tool to ensure a clean structured JSON-like dict response
        res = self.client.call_tool("sandbox_file_operations", {
            "action": "read",
            "path": "/does/not/exist.txt"
        })
        inner = self.client._parse_inner_response(res)
        
        # Assert: it returns a structured JSON-like failure
        self.assertIsInstance(inner, dict)
        self.assertFalse(inner.get("success", True))
        self.assertIn("error", inner)

        # Action 2: Run a malformed command
        success, output, error = self.client.exec_shell("non_existent_command_12345")
        
        # Assert: it should cleanly fail returning success = False and error message
        self.assertFalse(success)
        self.assertTrue(len(error) > 0)

if __name__ == "__main__":
    unittest.main()

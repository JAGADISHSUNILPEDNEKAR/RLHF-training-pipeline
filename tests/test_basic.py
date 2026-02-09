import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestImports(unittest.TestCase):
    def test_config_import(self):
        try:
            from src.config import config
            self.assertIsNotNone(config)
        except ImportError:
            self.fail("Could not import config")

    def test_utils_import(self):
        try:
            from src.utils import set_seed
        except ImportError:
            self.fail("Could not import utils")

if __name__ == '__main__':
    unittest.main()

## libraries
import os
import sys
import json
import unittest

## project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## modules
from src.data.loaders.bitcoin import BitcoinProcessor

## test case
class TestBitcoinProcessor(unittest.TestCase):

    ## set up test case
    def setUp(self):
        """ Sets up the test environment. """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.root_path = os.path.join(project_root, 'data', 'original')
        self.processor = BitcoinProcessor(root_path=self.root_path + os.path.sep, index=10)

    ## test processor
    def test_run_pipeline(self):
        """ Tests the full pipeline execution. """
        results = self.processor.run()

        ## check if the result is a dictionary
        self.assertIsInstance(results, dict)

        ## check for the presence of 'invariants' and 'events' keys
        self.assertIn("invariants", results)
        self.assertIn("events", results)

        ## check the type of 'invariants'
        self.assertIsInstance(results["invariants"], dict)

        ## check if 'events' is a valid json string
        try:
            json.loads(results["events"])
        except json.JSONDecodeError:
            self.fail("'events' is not a valid JSON string.")

if __name__ == '__main__':
    unittest.main()

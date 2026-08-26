#import pytest


#def test_code_is_tested():
 #   assert False


import unittest
from pathlib import Path
import tempfile

import pandas as pd

from mlops_fraudulent_transactions.dataset import CreditCardDataset


class TestCreditCardDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.csv_path = Path(self.temp_dir.name) / "creditcard.csv"
        self.output_path = Path(self.temp_dir.name) / "processed" / "dataset.csv"

        self.data = pd.DataFrame({
            "Time": [100, 200, 300],
            "V1": [0.1, 0.2, 0.3],
            "V2": [1.1, 1.2, 1.3],
            "Amount": [100.0, 200.0, 300.0],
            "Class": [0, 0, 1],
        })

        self.data.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_returns_dataframe(self):
        dataset = CreditCardDataset(self.csv_path)

        result = dataset.load()

        self.assertIsInstance(result, pd.DataFrame)

    def test_load_returns_correct_number_of_rows(self):
        dataset = CreditCardDataset(self.csv_path)

        result = dataset.load()

        self.assertEqual(len(result), 3)

    def test_load_returns_expected_columns(self):
        dataset = CreditCardDataset(self.csv_path)

        result = dataset.load()

        self.assertListEqual(
            list(result.columns),
            ["Time", "V1", "V2", "Amount", "Class"]
        )

    def test_save_without_load_raises_error(self):
        dataset = CreditCardDataset(self.csv_path)

        with self.assertRaises(RuntimeError):
            dataset.save(self.output_path)

    def test_save_creates_file(self):
        dataset = CreditCardDataset(self.csv_path)
        dataset.load()

        dataset.save(self.output_path)

        self.assertTrue(self.output_path.exists())

    def test_saved_file_contains_same_data(self):
        dataset = CreditCardDataset(self.csv_path)
        dataset.load()

        dataset.save(self.output_path)

        result = pd.read_csv(self.output_path)

        pd.testing.assert_frame_equal(
            self.data,
            result
        )


if __name__ == "__main__":
    unittest.main()
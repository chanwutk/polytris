import unittest

from reproduce.selection import select_ids


class ArtifactSelectionTest(unittest.TestCase):
    def setUp(self):
        self.grid = {
            'pruned': {'tracking_accuracy_threshold': 0.8},
            'unpruned': {'tracking_accuracy_threshold': None},
        }
        self.report = {
            'dataset': 'caldot1-y05',
            'split': 'valid',
            'configurations': [
                {'variant_id': name, 'config': config.copy(), 'pareto': name == 'pruned'}
                for name, config in self.grid.items()
            ],
        }

    def test_defaults_to_pareto_front_and_accepts_an_explicit_variant(self):
        self.assertEqual(select_ids(self.report, 'caldot1-y05', None, self.grid), ['pruned'])
        self.assertEqual(select_ids(self.report, 'caldot1-y05', 'unpruned', self.grid), ['unpruned'])

    def test_rejects_wrong_dataset_and_unknown_variant(self):
        with self.assertRaisesRegex(ValueError, 'Expected validation results'):
            select_ids(self.report, 'jnc0', None, self.grid)
        with self.assertRaisesRegex(ValueError, 'absent from the validation results'):
            select_ids(self.report, 'caldot1-y05', 'unknown', self.grid)

    def test_rejects_config_that_disagrees_with_the_current_grid(self):
        self.report['configurations'][0]['config']['tracking_accuracy_threshold'] = 0.4
        with self.assertRaisesRegex(ValueError, 'does not match the current sweep grid'):
            select_ids(self.report, 'caldot1-y05', None, self.grid)

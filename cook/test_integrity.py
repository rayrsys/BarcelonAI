"""Tests for errors that would invalidate the experiment or submission."""
import unittest
import numpy as np
from cook.external import parse, corrupt, prepare
from cook.model import design, swap_features, decode, score

class Integrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f, cls.y = prepare('J03WPY')

    def test_target_never_enters_features(self):
        self.assertNotIn('target', self.f)
        _, names = design(self.f)
        self.assertFalse(any('label' in s or 'target' in s for s in names))

    def test_corruption_independent_of_labels(self):
        d = parse('J03WPY')
        # Small slice retains both half IDs and avoids allocating full-match noise twice.
        tiny = {k:(v[:200].copy() if k in ('players','ball','half','target') else v.copy()) for k,v in d.items()}
        tiny['half'][100:] = 2
        a = corrupt(tiny,99)
        tiny['target'] = (tiny['target'] + 1) % 3
        b = corrupt(tiny,99)
        np.testing.assert_equal(a['players'],b['players'])
        np.testing.assert_equal(a['ball'],b['ball'])

    def test_team_swap_is_involution(self):
        twice = swap_features(swap_features(self.f))
        for k in self.f:
            np.testing.assert_allclose(twice[k],self.f[k],atol=1e-6,equal_nan=True)

    def test_decoder_resets_at_halftime(self):
        p = np.array([[.99,.005,.005]]*30 + [[.005,.005,.99]] + [[.005,.99,.005]]*30)
        h = np.array([1]*30+[0]+[2]*30)
        T = np.eye(3)*.998 + (1-np.eye(3))*.001
        lab = decode(p,h,T)
        self.assertTrue(np.all(lab[:30]=='H'))
        self.assertEqual(lab[30],'X')
        self.assertTrue(np.all(lab[31:]=='A'))

    def test_external_label_mapping_and_metric(self):
        self.assertEqual(set(np.unique(self.y)),{-1,0,1,2})
        lab = np.array(['X','H','A','D'])[self.y+1]
        result = score(self.y,lab)
        self.assertEqual(result['spell_f1_mean'],1.)
        self.assertEqual(result['macro_f1'],1.)

if __name__ == '__main__':
    unittest.main()

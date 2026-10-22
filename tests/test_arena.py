import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from arena import evaluate,samples

class ArenaTests(unittest.TestCase):
 def test_generation_is_deterministic_and_balanced(self):
  self.assertEqual(samples(2,'source',20),samples(2,'source',20))
  _,y=samples(2,'target',20); self.assertEqual(sum(y),10)
 def test_experiment_reports_both_domains_for_all_methods(self):
  out=evaluate(1); self.assertEqual(len(out['raw']),8)
  self.assertEqual({x['method'] for x in out['summary']},{'source_only','target_only_adaptation','source_target_replay','frozen_source_plus_sparse_residual'})
  self.assertTrue(all(0<=x['accuracy_mean']<=1 for x in out['summary']))

if __name__=='__main__': unittest.main()

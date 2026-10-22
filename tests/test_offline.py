"""Small source-level checks without training, GPU access or cloud calls."""
import ast
import contextlib
import copy
import importlib.util
import io
import random
import unittest
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


class OfflineChecks(unittest.TestCase):
    def test_all_experiment_python_sources_parse(self):
        paths = [p for p in ROOT.glob('lab*/**/*.py') if '__pycache__' not in p.parts]
        self.assertEqual(len(paths), 54)
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text('utf-8-sig'), filename=str(path))

    def test_binary_search_present_and_absent(self):
        fn = load('lab1/binary_search.py', 'binary_search').binary_search
        self.assertEqual(fn([1, 3, 5, 7], 5), 2)
        self.assertEqual(fn([1, 3, 5, 7], 4), -1)

    def test_binary_search_empty_and_duplicate(self):
        fn = load('lab1/binary_search.py', 'binary_search_duplicate').binary_search
        self.assertEqual(fn([], 1), -1)
        self.assertEqual(fn([1, 3, 3, 3, 5], 3), 3)

    def test_matrix_add_does_not_mutate_input(self):
        fn = load('lab1/matrix.py', 'matrix').matrix_add
        left, right = [[1, 2], [3, 4]], [[4, 3], [2, 1]]
        old_left, old_right = copy.deepcopy(left), copy.deepcopy(right)
        self.assertEqual(fn(left, right), [[5, 5], [5, 5]])
        self.assertEqual((left, right), (old_left, old_right))

    def test_a_star_solved_state(self):
        mod = load('lab3/15-puzzle_a.py', 'puzzle_astar_solved')
        end, parents = mod.A_star(mod.target, mod.h1)
        self.assertEqual(end, mod.target)
        self.assertIsNone(parents[end][0])

    def test_a_star_one_move(self):
        mod = load('lab3/15-puzzle_a.py', 'puzzle_astar_one')
        start = tuple(range(1, 15)) + (0, 15)
        end, parents = mod.A_star(start, mod.h1)
        self.assertEqual(end, mod.target)
        self.assertEqual(parents[end], (start, 15))

    def test_ida_star_one_move(self):
        mod = load('lab3/15-puzzle-ida.py', 'puzzle_idastar')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            mod.IDA_star(tuple(range(1, 15)) + (0, 15), mod.h1)
        self.assertIn('[15]', output.getvalue())

    def replay_buffer_class(self):
        # Load the original class without importing torch or constructing an agent.
        path = ROOT / 'lab6/DQN/agent_dir/agent_dqn.py'
        tree = ast.parse(path.read_text('utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ReplayBuffer')
        namespace = {'deque': deque, 'random': random}
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(path), 'exec'), namespace)
        return namespace['ReplayBuffer']

    def test_n_step_return_and_terminal_flush(self):
        buffer = self.replay_buffer_class()(10, 3, 0.5)
        buffer.push(0, 0, 1, 1, False)
        buffer.push(1, 0, 2, 2, False)
        self.assertEqual(len(buffer), 0)
        buffer.push(2, 0, 4, 3, True)
        self.assertEqual(len(buffer), 3)
        self.assertEqual([t[2] for t in buffer.buffer], [3.0, 4.0, 4.0])
        self.assertTrue(all(t[3] == 3 and t[4] for t in buffer.buffer))

    def test_short_episode_and_bounded_capacity(self):
        buffer = self.replay_buffer_class()(2, 3, 0.5)
        buffer.push(0, 0, 1, 1, False)
        buffer.push(1, 0, 2, 2, True)
        self.assertEqual(len(buffer), 2)
        self.assertEqual([t[2] for t in buffer.buffer], [2.0, 2.0])
        buffer.push(9, 0, 8, 10, True)
        self.assertEqual(len(buffer), 2)
        self.assertEqual(buffer.buffer[-1], (9, 0, 8.0, 10, True))

    def test_model_credentials_are_read_from_environment(self):
        path = ROOT / 'lab7/agents/src/config.py'
        tree = ast.parse(path.read_text('utf-8-sig'))
        fields = {n.target.id: n.value for n in ast.walk(tree)
                  if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        for name in ['api_key', 'vision_api_key', 'embedding_api_key']:
            with self.subTest(field=name):
                self.assertFalse(isinstance(fields[name], ast.Constant) and fields[name].value)


if __name__ == '__main__':
    unittest.main()

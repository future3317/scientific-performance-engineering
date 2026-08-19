from pathlib import Path
import importlib.util

def test_solution_api():
    p=Path(__file__).parents[1]/'workspace'/'solution.py'; s=importlib.util.spec_from_file_location('s',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert callable(m.run_episode_task)

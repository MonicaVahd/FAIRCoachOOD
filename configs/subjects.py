"""
configs/subjects.py -- read the EC3D subject list from configs/ec3d_subjects.yaml.

The real subject names are NOT stored in this repository.  Copy
configs/ec3d_subjects.example.yaml to configs/ec3d_subjects.yaml (git-ignored)
and replace S1..S4 with the names found in your copy of the EC3D pickle.
The order of the list defines the LOSO round numbers.

A minimal parser is used so that no extra dependency (PyYAML) is required.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(_HERE, "ec3d_subjects.yaml")


def load_ec3d_subjects(path=None):
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.  Copy configs/ec3d_subjects.example.yaml to "
            "configs/ec3d_subjects.yaml and enter the subject names of your EC3D copy."
        )
    subjects, in_list = [], False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.strip() == "subjects:":
                in_list = True
                continue
            if in_list and line.lstrip().startswith("- "):
                subjects.append(line.lstrip()[2:].strip().strip("'\""))
            elif in_list and not line.startswith((" ", "\t")):
                break
    if not subjects:
        raise ValueError(f"No 'subjects:' list found in {path}")
    return subjects

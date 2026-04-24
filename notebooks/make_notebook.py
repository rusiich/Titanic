from pathlib import Path
import nbformat as nbf
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

sys.path.append(str(PROJECT_ROOT))

from configs import config


PROJECT_ROOT = Path(config.paths.path_to_project)
FILES = [
    "configs.py",
    "src/data.py",
    "src/features.py",
    "src/model.py",
    "src/train.py",
    "main.py"
]

def read_requirements(path: Path) -> list[str]:
    packages = []

    if not path.exists():
        print(f"requirements.txt не найден: {path}")
        return packages

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        # пропускаем пустые строки и комментарии
        if not line or line.startswith("#"):
            continue

        packages.append(line)

    return packages

packages = read_requirements(REQUIREMENTS_PATH)

nb = nbf.v4.new_notebook()
cells = []

if packages:
    pip_command = "!pip install " + " ".join(packages) + " -q"
else:
    pip_command = "# requirements.txt пустой или не найден"
cells.append(
    nbf.v4.new_code_cell(pip_command)
)

for file in FILES:
    path = PROJECT_ROOT / file
    code = path.read_text(encoding="utf-8")

    cells.append(nbf.v4.new_markdown_cell(f"## `{file}`"))
    cells.append(nbf.v4.new_code_cell(code))

nb["cells"] = cells

with open("notebooks/project_all_in_one.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
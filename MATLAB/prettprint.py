# prettyjson.py
import json, pathlib, shutil, tempfile

def prettyprint_file(path):
    """
    Re-write *path* (a single .json file) with 2-space indentation.
    Returns True on success.
    """
    p = pathlib.Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    tmp  = tempfile.NamedTemporaryFile(delete=False, dir=p.parent, suffix=".tmp")
    with open(tmp.name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    shutil.move(tmp.name, p)
    return True

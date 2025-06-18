import nbformat, types
from nbconvert import PythonExporter
import pandas as pd

def load_ipynb(path, module_name="nb_module"):
    nb = nbformat.read(path, as_version=4)
    source, _ = PythonExporter().from_notebook_node(nb)
    mod = types.ModuleType(module_name)
    exec(source, mod.__dict__)
    return mod

def main():
    # Load the notebook as a live module
    nb = load_ipynb("valorant_model.ipynb", "valorant_model")

    # Verify the function exists
    if hasattr(nb, "get_kill_prediction"):
        print("get_kill_prediction loaded successfully.")
    else:
        raise AttributeError("get_kill_prediction not found in the notebook.")

    # Test example
    test = pd.DataFrame(
        [{
            "player_name": "johnqt",
            "team": "Sentinels",
            "opponent_team": "FNATIC",
            "stat_type": "maps_1-2_kills",
        }]
    )

    test["predicted_kills"] = test.apply(nb.get_kill_prediction, axis=1)

    # Nicely formatted console output
    print("\n=== Prediction Result ===")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
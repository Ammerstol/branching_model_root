
import json
from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PARAMETERS_DIR = EXPERIMENTS_DIR / "parameters"
RESULTS_DIR = EXPERIMENTS_DIR / "results"


def load_experiment(experiment_name):
    """
    Loads an experiment's parameter file and prepares its results folder.

    Returns:
        config: Dictionary containing all experiment parameters.
        output_dir: Path to the experiment's results directory.

    A snapshot of the parameters is saved in the results folder.
    Existing results cannot accidentally be mixed with a changed
    parameter configuration.
    """

    if Path(experiment_name).name != experiment_name:
        raise ValueError("Provide an experiment name, not a file path.")

    config_path = PARAMETERS_DIR / f"{experiment_name}.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment configuration not found: {config_path}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    if config.get("experiment_name") != experiment_name:
        raise ValueError(
            "The experiment_name inside the JSON does not match "
            "the parameter filename."
        )

    output_dir = RESULTS_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a snapshot of the configuration used for this experiment.
    snapshot_path = output_dir / "parameters.json"

    if snapshot_path.exists():

        with open(snapshot_path, "r") as f:
            previous_config = json.load(f)

        if previous_config != config:
            raise ValueError(
                f"Parameters for {experiment_name} have changed "
                "since the results folder was created. "
                "Use a new experiment name to avoid mixing results."
            )

    else:

        with open(snapshot_path, "w") as f:
            json.dump(config, f, indent=4)

    return config, output_dir

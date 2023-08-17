import argparse
import pathlib
from src.utils.similarity_judgment.analysis import run_standard_analysis_one_layer
import papermill as pm
import glob
from tqdm import tqdm

dataframe_path = "results/similarity_judgments/contour_completion/dataframe.csv"


DEFAULT = {"dataframe_path": None}


def generate_report(dataframe_path=None):
    if dataframe_path is None:
        all_dataframes_files = glob.glob(
            "results/similarity_judgments/**/*dataframe.csv", recursive=True
        )
    else:
        all_dataframes_files = [dataframe_path]

    for df_path in tqdm(all_dataframes_files):
        parameters = {
            "dataframe_path": df_path,
            "idx_layer_used": -1,
        }

        report_folder = pathlib.Path(df_path).parent / "report"
        report_folder.mkdir(exist_ok=True, parents=True)

        pm.execute_notebook(
            "src/utils/similarity_judgment/report_template.ipynb",
            str(report_folder / "report.ipynb"),
            parameters=parameters,
            progress_bar=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataframe_path",
        "-dfp",
        default=DEFAULT["dataframe_path"],
        help="The path to the dataframe that we want to generate the report of. If not given, we will look for all */dataframe.csv files in the results/similarity_judgments folder",
    )

    args = parser.parse_known_args()[0]
    generate_report(**args.__dict__)

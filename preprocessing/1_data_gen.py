################################################################################
######################### Import Requisite Libraries ###########################
import typer
import pandas as pd
import os
import sys
from pathlib import Path
from eda_toolkit import add_ids

sys.path.append(str(Path(__file__).resolve().parents[1]))

################################################################################

from core.constants import var_index

app = typer.Typer()

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")


@app.command()
def main(
    input_data_file: str = "./data/raw/NUFORC_DATA.xlsx",
    output_data_file: str = "./data/raw/nuforc_data.parquet",
):
    """
    Converts input data file to parquet format.
    Handles both CSV and Parquet inputs.

    Args:
        input_data_file (str): Path to the input file (csv or parquet).
        output_data_file (str): Path to save the output parquet file.
    """

    input_path = Path(input_data_file)
    output_path = Path(output_data_file)

    ############################################################################
    # Step 1. Check if output already exists and is same as input
    ############################################################################

    if output_path.exists() and input_path.suffix == ".parquet":
        if input_path.resolve() == output_path.resolve():
            print(f"Input file is already a parquet at target location: {output_path}")
            print("Skipping conversion.")
            return

    ############################################################################
    # Step 2. Read the input data file based on extension
    ############################################################################

    print(f"Reading input file: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_data_file)
        print("Loaded parquet file")
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_data_file)
        print("Loaded CSV file")
    elif input_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(input_data_file)
        print("Loaded Excel file")
    else:
        raise ValueError(
            f"Unsupported file format: {input_path.suffix}. Use .csv, .parquet, or .xlsx"
        )

    ############################################################################
    # Step 3. Generate Index and Set It
    ############################################################################

    # Add a column of unique IDs with 9 digits and call it "census_id"
    df = add_ids(
        df=df,
        id_colname=var_index,
        num_digits=9,
        seed=111,
        set_as_index=True,
    )

    try:
        df.set_index(var_index, inplace=True)
    except KeyError:
        print(f"Warning: '{var_index}' column not found in dataframe")
    except:
        print("Index already set or error setting index")

    print(f"\nInput Data Shape: {df.shape}")
    print(f"Unique indices: {df.index.unique().shape[0]}")

    ############################################################################
    # Step 4. Save to parquet
    ############################################################################

    df["Summary"]  = df["Summary"].astype(str)
    df["State"]    = df["State"].astype(str)
    df["Occurred"] = pd.to_datetime(df["Occurred"], errors="coerce")
    df["Reported"] = pd.to_datetime(df["Reported"], errors="coerce")
    print(f"\n{df.head()}")

    df.to_parquet(output_data_file)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    app()

import pandas as pd
import numpy as np

INPUT_MESSAGE = "Enter the name (with .csv ending) of the csv file to be graded:"
CSV_ERROR = "The provided file name is invalid"
TITLE_ERROR = "The title of the columns should be 'PassengerId' and 'Survived', in this order"
COL1_ERROR = "The PassengerId column must have values starting at 892 and ending at 1309"
SHAPE_ERROR = "The shape of the file is incorrect, ensure that you have added titles"
END_MESSAGE = "Please rerun the grader with the required changes"
expected_op_file = "output.csv"


def main():
    file_name = input(INPUT_MESSAGE)
    try:
        df = pd.read_csv(file_name)
        df_exp = pd.read_csv(expected_op_file)
        if not df.shape == df_exp.shape:
            print(SHAPE_ERROR)
            print(END_MESSAGE)
            return
        if not np.array_equal(df.columns, df_exp.columns):
            print(TITLE_ERROR)
            print(END_MESSAGE)
            return
        if not np.array_equal(df["PassengerId"], df_exp["PassengerId"]):
            print(COL1_ERROR)
            print(END_MESSAGE)
            return
        correct = df_exp["Survived"] == df["Survived"]
        accuracy = (100 * np.sum(correct)) / len(correct)
        accuracy = round(accuracy, 3)
        print("Your prediction has an accuracy of", accuracy, "%")
    except:
        print(CSV_ERROR)
        print(END_MESSAGE)
        return

if __name__ == '__main__':
    main()

import pandas as pd
import argparse
import os

def grade():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Give the path of the file to be graded")
    args = parser.parse_args()
    data_to_be_graded = pd.read_csv(os.path.join(os.getcwd(), args.path))
    expected = pd.read_csv('./output.csv')
    accuracy = sum(expected['Survived'].values == data_to_be_graded['Survived'].values) / len(expected)
    print("Your accuracy is {}%".format(round(accuracy * 100, 2)))

if __name__ == '__main__':
    grade()
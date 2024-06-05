# prefix = 'dataset/wikiart/'
import argparse
import os

import pandas as pd

folders = [
    "Abstract_Expressionism",
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau",
    "Baroque",
    "Color_Field_Painting",
    "Contemporary_Realism",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Impressionism",
    "Mannerism_Late_Renaissance",
    "Minimalism",
    "Naive_Art_Primitivism",
    "New_Realism",
    "Northern_Renaissance",
    "Pointillism",
    "Pop_Art",
    "Post_Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Synthetic_Cubism",
    "Ukiyo_e",
]

for folder in folders:
    os.makedirs("dataset/style/train/" + folder, exist_ok=True)
    os.makedirs("dataset/style/test/" + folder, exist_ok=True)

if not os.path.exists("dataset/wikiart"):
    print('Please download the style dataset, and put in "dataset/wikiart"')
    exit(0)

absolute_path = os.path.abspath("dataset/wikiart/")
target_path = os.path.abspath("dataset/style/train/")
# cnt = 0

# if __name__ == "__main__":
parser = argparse.ArgumentParser(description="Training configuration")
parser.add_argument(
    "--train_csv",
    type=str,
    default="dataset/wikiart_csv/style_train.csv",
    help="path to the csv file for style train images (default: ./dataset/wikiart_csv/style_train.csv)",
)

parser.add_argument(
    "--test_csv",
    type=str,
    default="dataset/wikiart_csv/style_test.csv",
    help="path to the csv file for style test images (default: ./dataset/wikiart_csv/style_test.csv)",
)

train_data = pd.read_csv("dataset/wikiart_csv/style_train.csv", header=None)
train_data.columns = ["path", "style"]

test_data = pd.read_csv("dataset/wikiart_csv/style_test.csv", header=None)
test_data.columns = ["path", "style"]

for i, row in train_data.iterrows():
    # move file to target path
    # print(os.path.join(absolute_path, row['path']))
    if os.path.exists(os.path.join(absolute_path, row["path"])):
        os.rename(
            os.path.join(absolute_path, row["path"]),
            os.path.join(target_path, row["path"]),
        )

target_path = os.path.abspath("dataset/style/test/")
for i, row in test_data.iterrows():
    # move file to target path
    # print(os.path.join(absolute_path, row['path']))
    if os.path.exists(os.path.join(absolute_path, row["path"])):
        os.rename(
            os.path.join(absolute_path, row["path"]),
            os.path.join(target_path, row["path"]),
        )

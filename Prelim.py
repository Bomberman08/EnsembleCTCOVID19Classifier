import shutil
import os.path
import splitfolders
import pip


def install_libs():
    pip.main(["install","pytest-shutil","tensorflow","scikit-learn","split-folders","keras_preprocessing","matplotlib","seaborn","keras","numpy","tk","Pillow"])

def split_dataset():
    if os.path.exists("Dataset"):
      if(os.path.exists("SplitDataset")):
          shutil.rmtree("SplitDataset")
      print("Splitting Dataset...")
      splitfolders.ratio('Dataset', output="SplitDataset", seed=1337, ratio=(0.7, 0.15, 0.15))
      print("Split Complete. The Split Dataset is stored inside the SplitDataset folder")
    else:
        print("Dataset to be split cannot be found. Please move the dataset into this program folder and rename as 'Dataset'")

if __name__ == '__main__':
    install_libs()
    split_dataset()







import sys
import importlib

sys.path.insert(0, r"C:\Users\ARNAV\Desktop\BM_DeeplearningModel\pneumonia-detection")

mod = importlib.import_module("src.data_loader")
mod.main()

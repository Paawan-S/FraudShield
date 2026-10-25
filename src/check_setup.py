import sys
import pandas
import numpy
import sklearn
import imblearn
import xgboost
import shap
import matplotlib
import seaborn

print("Python:", sys.version.split()[0])

libraries = [pandas, numpy, sklearn, imblearn, xgboost, shap, matplotlib, seaborn]
for lib in libraries:
    print(f"{lib.__name__:<12} {lib.__version__}")

print("\nSetup OK: all libraries imported successfully.")
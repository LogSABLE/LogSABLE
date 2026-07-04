import os, re, math, json, random, argparse, time
from typing import List, Tuple, Dict, Any
import numpy as np
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import json
from dataclasses import dataclass
import sqlite3, pickle, io, time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import os, yaml, torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import torch
import torch.nn as nn


WINDOW   = int(os.environ.get("WINDOW", 20))
STRIDE   = int(os.environ.get("STRIDE", 1))
VAL_SIZE = float(os.environ.get("VAL_SIZE", 0.20))
TEST_SIZE= float(os.environ.get("TEST_SIZE", 0.30))
BATCH    = int(os.environ.get("BATCH_SIZE", 128))
LR       = float(os.environ.get("LR", 1e-3))
EPOCHS   = int(os.environ.get("EPOCHS", 15))  
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
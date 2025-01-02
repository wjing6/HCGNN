import torch
import sys
import prepare_data

sys.path.append("..")
from lib.prepare_data import edgeList
# root_folder_path = "/data01/liuyibo/byte_1/graph_data/edge_tables"

# return_tensor = prepare_data.load_from_edge_list(root_folder_path)
# file_path = './b.edgelist'
# with open(file_path, "rb") as f:
#     line = f.readline()
#     step = 1
#     while (line) and step < 100:
#         print (line)
#         line = f.readline()
#         step += 1
# feature_dimensions = 128
# edgelist = edgeList('./b.edgelist')
# edge_index, max_vid = edgelist.getEdgeIndex()
# prepare_data.generating_feature_random(max_vid + 1, 128)

# print ("=" * 20 + "finish" + "=" * 20)

print ("loading")
feature_tensor = torch.load("x.zip")
print (feature_tensor.shape)
print (feature_tensor[0])
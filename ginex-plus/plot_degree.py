import argparse
import matplotlib.pyplot as plt
import json
import os
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
args = argparser.parse_args()

def plot_degree(degree_path):
    xlabel = "Out Degree"
    figname = "fig/{}_sample_degree.{}".format(args.dataset, "png")
    degree_list = torch.load(degree_path)
    degree_list = degree_list.tolist()
    # for idx, degree in enumerate(degree_list):
    #     # access_rtime_list stores N objects, each object has one access pattern list
    #     plt.scatter(degree, idx, s=8)
    plt.barh(range(len(degree_list)), degree_list)
    plt.xlabel(xlabel)
    plt.ylabel("Sampled object")  # (sorted by first access time)
    plt.savefig(figname, bbox_inches="tight")
    plt.clf()

if __name__ == '__main__':
    root = "/data01/liuyibo/"
    dataset_folder = os.path.join(root, args.dataset)
    degree_path = os.path.join(dataset_folder, args.dataset + '.degree')
    plot_degree(degree_path)
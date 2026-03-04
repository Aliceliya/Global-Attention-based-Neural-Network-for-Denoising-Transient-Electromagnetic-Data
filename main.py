import random
import os
import numpy as np

from make_data import make_inv_tem_data, make_for_tem_data
from make_model import make_for_model, make_inv_model




def main_make_for_data(data_number,max_num_layer, start_time, end_time):
    model = ['high', 'medium', 'low']
    layer = np.linspace(5, max_num_layer, max_num_layer-4, dtype=int)
    make_for_tem_data(data_number=data_number, start_time=start_time, end_time=end_time, data_len=31, source_type=7,
                      receiver_type=3, time_model='logspace', constraint_model=1, constraint_reweighting=0,
                      num_layers=layer, con_model=model)



def forward(data_number):
    current_path = os.path.abspath(__file__)
    parent_dir = os.path.abspath(os.path.join(current_path, os.pardir))
    grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
    for i in range(data_number):
        os.chdir(grandparent_dir+f"/data/for/data_{str(i+1).zfill(6)}")
        exe_file = os.path.join(grandparent_dir, "AarhusInv_intel_2013.exe")
        data_file = os.path.join(grandparent_dir, f"data/for/data_{str(i+1).zfill(6)}", "model.mod")
        con_file = os.path.join(grandparent_dir, "AarhusInv.con")
        command = f'wine {exe_file} {data_file} {con_file}'
        os.system(command)


def main_make_inv_data(data_number):
    Layer = []
    Thick = []
    for i in range(data_number):
        path = f"../data/inv/data_{str(i+1).zfill(6)}/model.mod"
        with open(path, 'r') as f:
            lines = f.readlines()
            x = lines[5].split()[0]
            x = float(x)
            layer, thickness = make_inv_model(con_v=100)
            Layer.append(layer)
            Thick.append(thickness)
    make_inv_tem_data(data_number=data_number, inv_layer=Layer,
                      constraint_model=1, constraint_reweighting=0, thicknesses=Thick)


def inversion(data_number):
    current_path = os.path.abspath(__file__)
    parent_dir = os.path.abspath(os.path.join(current_path, os.pardir))
    grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
    for i in range(data_number):
        os.chdir(grandparent_dir+f"/data/inv/data_{str(i+1).zfill(6)}")
        exe_file = os.path.join(grandparent_dir, "AarhusInv_intel_2013.exe")


        data_file = os.path.join(grandparent_dir, f"data/inv/data_{str(i+1).zfill(6)}", "model.mod")
        con_file = os.path.join(grandparent_dir, "AarhusInv.con")
        command = f'wine {exe_file} {data_file} {con_file}'
        os.system(command)
#
main_make_for_data(data_number=1, max_num_layer=15, start_time=-5, end_time=-2)
forward(1)
# #
main_make_inv_data(1)
inversion(1)
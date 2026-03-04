import os
import shutil
import numpy as np
import random
from make_model import make_for_model, make_inv_model




def make_for_tem_data(data_number: int, start_time, end_time, data_len: int,
                  source_type: int, receiver_type: int, time_model: str,
                  constraint_model:int, constraint_reweighting: int, num_layers, con_model,
                  vertical_constraints=None, horizontal_constraints=None, reweighting_type=None,
                  inversion_parameter=1, max_iter=-1, prior_STD=-1,
                  transmitter_x=0.00, transmitter_y=0.00, transmitter_z=0.00,
                  receiver_x=0.00, receiver_y=0.00, receiver_z=0.00,
                  transmitter_1=40.00, transmitter_2=40.00,
                  data_transform_1=3, data_transform_2=3, data_transform_3=3,
                  waveform=1, wave_number=0,
                  cond_data=1e-3, std=1e-2, waveform_measure=0, filter_used=0):
    data_dir_ = "../data"

    for i in range(data_number):
        #  创建用于存放模型和场源信息的文件夹
        file_dir = f"{data_dir_}/for/data_{str(i+1).zfill(6)}"   # 创建文件夹路径
        # if os.path.exists(file_dir):  # 如果文件夹路径存在，则删除当前路径下的同名文件夹
        #     shutil.rmtree(file_dir)  # 删除操作
        # os.mkdir(file_dir,  0o755)  # 根据路径创建文件夹

        #  创建场源信息
        tem_name = f"tem"
        if time_model == 'logspace':
            time = np.logspace(start_time, end_time, data_len)  # 采样时间点，取对数时间
        else:
            time = np.linspace(start_time, end_time, data_len)  # 否则，取均匀间隔时间
        data_TEM = ""
        data_TEM += f"{tem_name}\n"  # 数据名字
        data_TEM += f"{source_type} {receiver_type}\n"  # 场源类型 接收机类型
        data_TEM += (f"{transmitter_x} {transmitter_y} {transmitter_z} "  # 发射机位置
                     f"{receiver_x} {receiver_y} {receiver_z}\n")  # 接收机位置
        data_TEM += f"{transmitter_1} {transmitter_2}\n"  # 发射机线圈属性
        data_TEM += f"{data_transform_1} {data_transform_2} {data_transform_3}\n"  # 输入数据类型， 反演过程数据类型， 输出数据类型(注：依赖波形）
        data_TEM += f"{waveform} {wave_number} 0\n"
        for j in range(len(time)):
            # 数据格式调整
            TIME = "{:e}".format(float("{:.10f}".format(time[j])))
            cond_data = "{:e}".format(float("{:.10f}".format(float(cond_data))))
            std = "{:e}".format(float("{:.10f}".format(float(std))))
            waveform = "{:e}".format(float("{:.10f}".format(float(waveform))))
            filter_used = "{:e}".format(float("{:.10f}".format(float(filter_used))))
            # 时间 正演基础数值(注：原本正演不需要，程序本身需要) 标准差  波形测量数据 滤波器数据(注：后两项在未设定波形时不需要)
            data = f"{TIME} {cond_data} {std} {waveform_measure} {filter_used}\n"
            data_TEM += data
        # 保存场源信息
        file = open(file_dir + f"/tem.tem", "w")
        file.write(data_TEM)
        file.close()

        ##创建模型信息
        model_name = f"model"
        data_model = ""
        data_model += f"{model_name}\n"
        data_model += f"{constraint_model} {constraint_reweighting} "  # 约束模型 约束重加权
        if vertical_constraints is not None:
            data_model += f"{vertical_constraints} "  # 垂直加权
        if horizontal_constraints is not None:
            data_model += f"{horizontal_constraints}"  # 水平加权
        if reweighting_type is not None:
            data_model += f"{reweighting_type}"  # 重新加权
        data_model += f"\n"
        data_model += f"1 {inversion_parameter} {tem_name}.tem\n"  # 正演/反演模型 tem数据名
        data_model += f"{max_iter}\n"  # 最大迭代次数
        # 创建随机的模型
        layers, thicknesses, con_layer = make_for_model(num_layer=15, con_model=random.choice(con_model))
        num_layer = len(layers)
        data_model += f"{num_layer}\n"  # 几层模型(注：模型定义隐含地球上方空气层，例如一个3层模型包含一个空气层，两个地面层和一个持续无限深度的底层)
        prior_STD = format(float("{:.10f}".format(float(prior_STD))))
        for j in range(num_layer):
            layer = format(float("{:.10f}".format(float(layers[j]))))
            data_model += f"{format(float(layer))} {prior_STD}\n"
        for j in range(num_layer-1):
            thickness = format(float("{:.10f}".format(float(thicknesses[j]))))
            data_model += f"{thickness} {prior_STD}\n"
        depth = 0
        for j in range(num_layer-1):
            depth += thicknesses[j]
            depths = float(depth)
            depths = format(float("{:.10f}".format(float(depths))))
            data_model += f"{depths} {prior_STD}\n"
        # 保存模型信息
        file = open(file_dir + f"/model.mod", "w")
        file.write(data_model)
        file.close()



# make_for_tem_data(2,0.001,1,1000,7,3, time_model='linspace',constraint_model=1,
#                   constraint_reweighting=0, layers=[30, 40, 50], thicknesses=[100, 100])


def make_inv_tem_data(data_number, inv_layer, constraint_model:int, constraint_reweighting: int, thicknesses,
                      vertical_constraints=None, horizontal_constraints=None,reweighting_type=None,
                      inversion_parameter=1, max_iter=50, prior_STD=0.0001):
    data_dir_ = "../data"
    for i in range(data_number):
        #  创建用于存放模型和场源信息的文件夹
        file_dir = f"{data_dir_}/inv/data_{str(i+1).zfill(6)}"   # 创建文件夹路径
        if os.path.exists(file_dir):  # 如果文件夹路径存在，则删除当前路径下的同名文件夹
            shutil.rmtree(file_dir)  # 删除操作
        os.mkdir(file_dir,  0o755)  # 根据路径创建文件夹
        # 读取正演信息
        file_path = f"{data_dir_}/for/data_{str(i+1).zfill(6)}/model00001.fwr"
        with open(file_path, 'r') as files:
            file_content = files.read()
        file = open(file_dir + f"/tem.tem", "w")
        file.write(file_content)
        file.close()

        model_name = f"model"
        data_model = ""
        data_model += f"{model_name}\n"
        data_model += f"{constraint_model} {constraint_reweighting} "  # 约束模型 约束重加权
        if vertical_constraints is not None:
            data_model += f"{vertical_constraints} "  # 垂直加权
        if horizontal_constraints is not None:
            data_model += f"{horizontal_constraints}"  # 水平加权
        if reweighting_type is not None:
            data_model += f"{reweighting_type}"  # 重新加权
        data_model += f"\n"
        data_model += f"1 {inversion_parameter} tem.tem\n"  # 正演/反演模型 tem数据名
        data_model += f"{max_iter}\n"  # 最大迭代次数

        Inv_layer, Thicknesses = inv_layer[i], thicknesses[i]
        num_layer = len(Inv_layer)
        data_model += f"{num_layer}\n"  # 几层模型(注：模型定义隐含地球上方空气层，例如一个3层模型包含一个空气层，两个地面层和一个持续无限深度的底层)
        prior_STD = format(float("{:.10f}".format(float(prior_STD))))
        for j in range(num_layer):
            layer = format(float("{:.10f}".format(float(Inv_layer[i]))))
            data_model += f"{format(float(layer))} {1}\n"
        for j in range(num_layer-1):
            thickness = format(float("{:.10f}".format(float(Thicknesses[j]))))
            data_model += f"{thickness} {prior_STD}\n"
        depth = 0
        for j in range(num_layer-1):
            depth += Thicknesses[j]
            depths = float(depth)
            depths = format(float("{:.10f}".format(float(depths))))
            data_model += f"{depths} {prior_STD}\n"
        # 保存模型信息
        file = open(file_dir + f"/model.mod", "w")
        file.write(data_model)
        file.close()

#
# make_inv_tem_data(data_number=2, inv_layer=[50, 50, 50, 50, 50, 50, 50, 50, 50],
# constraint_model=1,constraint_reweighting=0, thicknesses=[1, 1, 1, 1, 1, 1, 1, 1, 1])


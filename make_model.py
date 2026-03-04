import numpy as np
import random

def random_number(a, b, maximum):

    average = b // a
    minimum = 2 * average - maximum
    n_l = list()
    # n_l = [36, 39, 22, 43, 24, 97]
    for _ in range(0, a):
        n_l.append(random.randint(minimum, maximum))
    # n3_l = n_l.copy()
    s = 0
    for i in n_l:
        s = i + s
    # print(s)
    n2_l = list()
    if b - s > 0:
        n_l.sort(reverse=True)
        aa = int((b - s) / a)
        for i2 in n_l:
            if i2 + int((b - s) / a) <= maximum:
                n2_l.append(i2 + int((b - s) / a))
            else:
                n2_l.append(maximum)
                n_l[-1] = n_l[-1] + int((b - s) / a) - (maximum - i2)
                n_l.sort(reverse=True)
        n_l.sort()
        if (b - s) % a != 0:
            for i4 in range(0, (b - s) % a):
                if n2_l[i4] == maximum:
                    n_l.sort()
                    n2_l[-1] = n2_l[-1] + 1
                else:
                    n2_l[i4] = n2_l[i4] + 1
    else:
        n_l.sort()
        aa = int((s - b) / a)
        for i3 in n_l:
            if i3 - int((s - b) / a) >= minimum:
                n2_l.append(i3 - int((s - b) / a))
            else:
                n2_l.append(minimum)
                n_l[-1] = n_l[-1] - (int((s - b) / a) - (i3 - minimum))
                n_l.sort(reverse=False)
        bb = (s - b) % a
        if (s - b) % a != 0:
            n2_l.sort(reverse=True)
            for i4 in range(0, (s - b) % a):
                if n2_l[i4] == minimum:
                    n2_l.sort(reverse=True)
                    n2_l[0] = n2_l[0] - 1
                else:
                    n2_l[i4] = n2_l[i4] - 1
    s2 = 0
    for i3 in n2_l:
        s2 = i3 + s2
    n2_l.sort(reverse=True)
    return n2_l

def make_for_model(num_layer, con_model):
    if con_model == 'high':
        con_layer = np.random.randint(500, 1000)
    elif con_model == 'medium':
        con_layer = np.random.randint(100, 500)
    else:
        con_layer = np.random.randint(10, 100)
    layer = np.ones(num_layer)
    layer = layer * con_layer

    thicknesses = random_number(20, 800, 160)

    for i in range(num_layer):
        if i == num_layer-1:
            pass
        elif i == 0:
            pass
        else:
            layer[i] = np.random.randint(1, 1000)
    return layer, thicknesses, con_layer


def make_inv_model(con_v, num_layer=50):

    Layer = np.ones(num_layer) * 600
    thickness = np.ones(num_layer) * 1
    return Layer, thickness




#!usr/bin/env python3
# -*-coding:utf8-*-

import socketserver
import cmath
import math
import threading
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from PIL import Image
import random
import queue
import time
import pickle
import socket
import configparser
import os
import time
import random

# elliott

Packet_number = 100
Line_number = 3
rootpath = ""
imgFlag = 16

cnt = 0  # 第几个包
fenge = 0 #每4000个包分割文件
num_of_line = 0  #有几条线波动
num_of_packet = 0  #有几个包波动
num_of_packet_group = 0  #有几组包波动，6组归0
now_path = ""        #当前路径

line_all = []
list_all = []
fig = plt.figure("Watch Window")
ax1 = fig.add_subplot(2, 1, 1, xlim=(0, Packet_number), ylim=(-21, 41))
ax2 = fig.add_subplot(2, 1, 2)
img = Image.open('./timg.jpg')
im = ax2.imshow(img)
ax2.axis('off')


for i in range(30):
    alist = list(range(0, Packet_number))
    list_all.append(alist)


class CSI():
    def __init__(self, Nrx, Ntx, rssi_a, rssi_b, rssi_c, noise, agc, csi):
        self.Nrx = Nrx
        self.Ntx = Ntx
        self.rssi_a = rssi_a
        self.rssi_b = rssi_b
        self.rssi_c = rssi_c
        self.noise = noise
        self.agc = agc
        self.csi = csi


class MyHandle(socketserver.BaseRequestHandler):
    def handle(self):
        tname = ""
        timestamp = ""
        while 1:
            conn = self.request
            print("收到连接")
            while 1:
                data = conn.recv(4096)
                name = "data"
                if len(data) <= 3 or data[2] != 187:
                    # print("nothing")
                    continue
                if (name != tname):
                    tname = name
                    timestamp = str(time.time())
                path = name
                global now_path
                now_path = path
                if (os.path.exists(path) is not True):
                    os.makedirs(path)
                name = name + timestamp
                self.deal_data(data, name)
            print("conn close...")
            conn.close()

    def translate(self, tmp):
        a = bin(tmp)
        a = a[-8:]
        b = ""
        if (a[0] == '0'):
            return int(a, 2)
        else:
            a = a[1:]
            for i in range(0, 7):
                if (a[i] == '0'):
                    b += '1'
                else:
                    b += '0'
        return 0 - int(b, 2) - 1

    def db(self, x):
        return 10 * math.log(x * x, 10)

    def dbinv(self, x):
        return 10 ** (x / 10)

    def get_total_rss(self, csi_st):
        rssi_mag = 0
        if csi_st.rssi_a != 0:
            rssi_mag = rssi_mag + self.dbinv(csi_st.rssi_a)
        if csi_st.rssi_b != 0:
            rssi_mag = rssi_mag + self.dbinv(csi_st.rssi_b)
        if csi_st.rssi_c != 0:
            rssi_mag = rssi_mag + self.dbinv(csi_st.rssi_c)
        return self.db(rssi_mag) / 2 - 44 - csi_st.agc

    def get_scaled_csi(self, csi_st):
        csi = csi_st.csi
        csi_sq = []
        csi_pwr = 0
        for x in csi:
            csi_sq.append(abs(x) ** 2)
            csi_pwr += abs(x) ** 2
        if (csi_pwr == 0):
            return -1
        rssi_pwr = self.dbinv(self.get_total_rss(csi_st))
        scale = rssi_pwr / (csi_pwr / 30)
        noise_db = csi_st.noise
        if (csi_st.noise == -127):
            noise_db = -92
        thermal_noise_pwr = self.dbinv(noise_db)
        quant_error_pwr = scale * (csi_st.Nrx * csi_st.Ntx)
        total_noise_pwr = thermal_noise_pwr + quant_error_pwr
        for i in range(0, len(csi)):
            csi[i] = csi[i] * ((scale / total_noise_pwr) ** 0.5)
        if csi_st.Ntx == 2:
            for i in range(0, len(csi)):
                csi[i] = csi[i] * (2 ** 0.5)
        elif csi_st.Ntx == 3:
            for i in range(0, len(csi)):
                csi[i] = csi[i] * (self.dbinv(4.5) ** 0.5)

    def deal_data(self, data, name):
        Nrx = int(data[11])
        Ntx = int(data[12])
        rssi_a = int(data[13])
        rssi_b = int(data[14])
        rssi_c = int(data[15])
        noise = self.translate(int(data[16]))
        agc = int(data[17])

        if Nrx * Ntx > 6:
            return
        payload = data[23:]
        csi = []
        index = 0
        for i in range(0, 30):
            index += 3
            remainder = index % 8
            for j in range(0, Nrx * Ntx):
                tmp1 = (payload[int(index / 8)] >> remainder) | (payload[int(index / 8) + 1] << (8 - remainder))
                tmp1 = self.translate(tmp1)
                tmp2 = (payload[int(index / 8) + 1] >> remainder) | (payload[int(index / 8) + 2] << (8 - remainder))
                tmp2 = self.translate(tmp2)
                csi.append(tmp1 + tmp2 * 1j)
                # print(str(tmp1)+"+"+str(tmp2)+"*i")
                index = index + 16
        csi_st = CSI(Nrx, Ntx, rssi_a, rssi_b, rssi_c, noise, agc, csi)
        if len(csi) == 180:
            csi_t = []
            for x in range(0, 180, 2):
                csi_t.append(csi[x])
            csi_st.csi = csi_t
        judge = self.get_scaled_csi(csi_st)
        if (judge == -1):
            return
        with open(name, 'ab+') as f:
            pickle.dump(csi_st.csi, f)
        global cnt
        global fenge
        global num_of_line
        global num_of_packet
        global num_of_packet_group
        for k in range(30):
            a = abs(csi_st.csi[k])
            if a == 0:
                a = 1
            list_all[k][cnt] = self.db(abs(a))
        if (cnt % 5 == 0):
            for o in range(30):
                sum = 0
                for u in range(5):
                    sum = sum + list_all[o][(cnt - u) % 90]
                avg = sum / (5 * 1.0)
                #global imgFlag
                if (abs(list_all[o][cnt] - avg) <= 3):
                    imgFlag = 16  # OK
                else:
                    num_of_line += 1
                    if num_of_line >= 10: 
                        num_of_packet = num_of_packet + 1
                        imgFlag = 10
            if num_of_packet >= 4:   
                imgFlag = 10  # warning
    

                num_of_packet = 0
                num_of_line = 0
            num_of_packet_group = (num_of_packet_group + 1) % 6
            if num_of_packet_group is 0:  
                num_of_packet = 0
                num_of_line = 0

        cnt = (cnt + 1) % Packet_number
        if cnt is 0:
            fenge = fenge + 1
        if fenge is 40:
            fenge = 0
            timestamp = str(time.time())
            global now_path
            name = now_path + "/" + timestamp

def init():
    for i in range(Line_number):
        line_all[i].set_data([], [])
    return line_all


# animation function.  this is called sequentially
def animate(i):
    x = list(range(0, Packet_number))
    for i in range(Line_number):
        y = list_all[i]
        line_all[i].set_data(x, y)
    global imgFlag
    if imgFlag > 15:
        img = Image.open('./timg.jpg')
    else:
        img = Image.open('./attention.jpg')
    im.set_array(img)
    return [im]


def server_thread():
    config = configparser.ConfigParser()
    config.read("conf.ini")
    ip = config.get("conf", "ip")
    port = config.get("conf", "port")
    global Packet_number, Line_number, rootpath
    Packet_number = int(config.get("conf", "Packet_number"))
    Line_number = int(config.get("conf", "Line_number"))
    rootpath = config.get("conf", "rootpath")
    s = socketserver.ThreadingTCPServer((ip, int(port)), MyHandle)
    s.serve_forever()


if __name__ == "__main__":

    # first set up the figure, the axis, and the plot element we want to animate
    for i in range(Line_number):
        line, = ax1.plot([], [], lw=1)  # the chuxi of string
        line_all.append(line)

    for j in range(Line_number):
        i = 0
        while i < Packet_number:
            list_all[j][i] = 0
            i = i + 1

    t = threading.Thread(target=server_thread)
    t.start()
    anim1 = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=1)
    plt.show()



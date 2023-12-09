#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 23:47:14 2023

@author: marc
"""

import matplotlib.pyplot as plt
import re
from math import log2

imgpath = "imgs/"

msglength = []
latency = []
compression = []
communication = []
decompression = []
reduce = []
detailedTimes = False

def readLists(dataset, function, machine, ngpus):
    if ngpus == 4:
        logfile = dataset+'/log-'+function+'_'+machine+'.log'
    else:
        logfile = dataset+'/log-'+function+'_'+machine+'_'+str(ngpus)+'GPUs.log'
    
    msglength = []
    latency = []
    compression = []
    communication = []
    decompression = []
    reduce = []
    detailedTimes = False
    
    with open(logfile, 'r') as file:
        for row in file:
            row = row.strip()
            if len(row) > 0:
                columns = re.split(r'\s+', row)
                print(len(columns), columns)
                if columns[0] != '#' and columns[0] != 'New' and columns[0] != 'Hi' and not columns[0].startswith('['):
                    if len(columns) > 3:
                        detailedTimes = True
                    print(len(columns), columns)
                    msglength.append(int(log2(int(columns[0]))))
                    #msglength.append(int(columns[0]) // 1024)
                    latency.append(float(columns[1]))
                    if detailedTimes:
                        compression.append(float(columns[2]))
                        communication.append(float(columns[3]))
                        decompression.append(float(columns[4]))
                        reduce.append(float(columns[5]))
    return [msglength, latency, compression, communication, decompression, reduce] 

# dataset  = 'msg_sppm'
dataset  = 'num_plasma'

function = 'allreduce'
# function = 'allreduce_orig'
# function = 'allreduce_woComp'
# function = 'bcast'
# function = 'bcast_orig'

machine   = 'funkel'
# machine  = 'jusuf'

[msg1, lat1, comp1, comm1, deco1, red1] = readLists(dataset, function, machine, 2)
[msg2, lat2, comp2, comm2, deco2, red2] = readLists(dataset, function, machine, 4)
if machine == 'jusuf':
    [msg3, lat3, comp3, comm3, deco3, red3] = readLists(dataset, function, machine, 8)

minidx = 12
maxidx = 23

fig1 = plt.figure()
plt.plot(msg2[minidx:maxidx], lat2[minidx:maxidx], marker='o', linestyle='-', color='blue', label='4 GPUs')
plt.plot(msg1[minidx:maxidx], lat1[minidx:maxidx], marker='s', linestyle='-', color='red', label='2 GPUs')
if machine == 'jusuf':
    plt.plot(msg3[minidx:maxidx], lat3[minidx:maxidx], marker='v', linestyle='-', color='green', label='8 GPUs')
plt.xlabel('log2(message size [Bytes])')
plt.ylabel('Time [µs]')
plt.title('Different CPU numbers for function ' + function + ' on ' + machine + ' (' + dataset + ')')
plt.legend()
plt.grid()
logfile  = 'log-'+function+'_'+machine
fig1.savefig(dataset+'/'+logfile+'_lineplot_nGPUS.png', dpi=300, bbox_inches='tight')

fig2 = plt.figure()
plt.plot(msg2[minidx:maxidx], lat2[minidx:maxidx], marker='o', linestyle='-', color='blue', label='4 GPUs')
plt.plot(msg1[minidx:maxidx], lat1[minidx:maxidx], marker='s', linestyle='-', color='red', label='2 GPUs')
if machine == 'jusuf':
    plt.plot(msg3[minidx:maxidx], lat3[minidx:maxidx], marker='v', linestyle='-', color='green', label='8 GPUs')
plt.xlabel('log2(message size [Bytes])')
plt.ylabel('Time [µs]')
plt.title('Different CPU numbers for function ' + function + ' on ' + machine + ' (' + dataset + ')')
plt.yscale('log')
plt.legend()
plt.grid()
logfile  = 'log-'+function+'_'+machine
fig2.savefig(dataset+'/'+logfile+'_lineplot-log_nGPUs.png', dpi=300, bbox_inches='tight')


# Show the plot
plt.show()

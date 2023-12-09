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

def readLists(dataset, function, machine):
    logfile = dataset+'/log-'+function+'_'+machine+'.log'
    
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
                    latency.append(float(columns[1]))
                    if detailedTimes:
                        compression.append(float(columns[2]))
                        communication.append(float(columns[3]))
                        decompression.append(float(columns[4]))
                        reduce.append(float(columns[5]))
    return [msglength, latency, compression, communication, decompression, reduce] 

# dataset  = 'msg_sppm'
dataset  = 'num_plasma'

# function1 = 'allreduce'
function1 = 'bcast'
# function2 = 'allreduce_orig'
function2 = 'bcast_orig'
function3 = ''
# function3 = 'allreduce_woComp'

# machine   = 'funkel'
machine  = 'jusuf'

[msg1, lat1, comp1, comm1, deco1, red1] = readLists(dataset, function1, machine)
[msg2, lat2, comp2, comm2, deco2, red2] = readLists(dataset, function2, machine)
if function3:
    [msg3, lat3, comp3, comm3, deco3, red3] = readLists(dataset, function3, machine)

minidx = 12
maxidx = 23

fig1 = plt.figure()
plt.plot(msg1[minidx:maxidx], lat1[minidx:maxidx], marker='o', linestyle='-', color='blue', label='butterfly with compression')
if function3:
    plt.plot(msg3[minidx:maxidx], lat3[minidx:maxidx], marker='v', linestyle='-', color='green', label='butterfly w/o compression')
plt.plot(msg2[minidx:maxidx], lat2[minidx:maxidx], marker='s', linestyle='-', color='red', label='original MPI_Allreduce')
plt.xlabel('log2(message size [Bytes])')
plt.ylabel('Time [µs]')
plt.title(function1 + ' vs ' + function2 + ' on ' + machine + ' (' + dataset + '), 4 GPUs')
plt.legend()
plt.grid()
logfile  = 'log-'+function1+'_'+machine
fig1.savefig(dataset+'/'+logfile+'_lineplot.png', dpi=300, bbox_inches='tight')

fig2 = plt.figure()
plt.plot(msg1[minidx:maxidx], lat1[minidx:maxidx], marker='o', linestyle='-', color='blue', label='butterfly with compression')
if function3:
    plt.plot(msg3[minidx:maxidx], lat3[minidx:maxidx], marker='v', linestyle='-', color='green', label='butterfly w/o compression')
plt.plot(msg2[minidx:maxidx], lat2[minidx:maxidx], marker='s', linestyle='-', color='red', label='original MPI_Allreduce')
plt.xlabel('log2(message size [Bytes])')
plt.ylabel('Time [µs]')
plt.title(function1 + ' vs ' + function2 + ' on ' + machine + ' (' + dataset + '), 4 GPUs')
plt.yscale('log')
plt.legend()
plt.grid()
logfile  = 'log-'+function1+'_'+machine
fig2.savefig(dataset+'/'+logfile+'_lineplot-log.png', dpi=300, bbox_inches='tight')

minidx = 12
maxidx = 22

if len(comp1) > 0:
    fig3 = plt.figure()
    plt.bar(msg1[minidx:maxidx], comp1[minidx:maxidx], label='compression', color='blue')
    plt.bar(msg1[minidx:maxidx], comm1[minidx:maxidx], label='communication', color='red', bottom=comp1[minidx:maxidx])
    plt.bar(msg1[minidx:maxidx], deco1[minidx:maxidx], label='decompression', color='green', 
            bottom=[comp1[i] + comm1[i] for i in range(minidx, maxidx)])
    plt.bar(msg1[minidx:maxidx], red1[minidx:maxidx], label='reduce', color='grey', 
            bottom=[comp1[i] + comm1[i] + deco1[i] for i in range(minidx, maxidx)])
    plt.xlabel('log2(message size [Bytes])')
    plt.ylabel('Time [µs]')
    plt.title(function1 + ' on ' + machine + ' (' + dataset + '), 4 GPUs')
    plt.legend()
    plt.grid()
    logfile  = 'log-'+function1+'_'+machine
    fig3.savefig(dataset+'/'+logfile+'_barplot-detail.png', dpi=300, bbox_inches='tight')
    
    fig4 = plt.figure()
    plt.bar(msg1[minidx:maxidx], [comp1[i] + deco1[i] + red1[i] for i in range(minidx, maxidx)], label='non-comm', color='blue')
    plt.bar(msg1[minidx:maxidx], comm1[minidx:maxidx], label='communication', color='red', 
            bottom= [comp1[i] + deco1[i] + red1[i] for i in range(minidx, maxidx)])
    plt.xlabel('log2(message size [Bytes])')
    plt.ylabel('Time [µs]')
    plt.title(function1 + ' on ' + machine + ' (' + dataset + '), 4 GPUs')
    plt.legend()
    plt.grid()
    logfile  = 'log-'+function1+'_'+machine
    fig4.savefig(dataset+'/'+logfile+'_barplot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


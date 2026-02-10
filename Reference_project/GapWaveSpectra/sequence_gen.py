# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:15:42 2023

@author: Mohamed Mossad

This is adapted from Oliver's comment in https://stackoverflow.com/a/60844608/10231656
It creates an array in a similar manner to seq function from R
"""

def seq(start, end, by = None, length_out = None):
   
    len_provided = True if (length_out is not None) else False
    by_provided = True if (by is not None) else False
    if (not by_provided) & (not len_provided):
        raise ValueError('At least by or length_out must be provided')
    width = end - start
    eps = pow(10.0, -14)
    if by_provided:
        if (abs(by) < eps):
            raise ValueError('by must be non-zero.')
    #Switch direction in case in start and end seems to have been switched (use sign of by to decide this behaviour)
        if start > end and by > 0:
            e = start
            start = end
            end = e
        elif start < end and by < 0:
            e = end
            end = start
            start = e
        absby = abs(by)
        if absby - width < eps: 
            length_out = int(width / absby)
        else: 
            #by is too great, we assume by is actually length_out
            length_out = int(by)
            by = width / (by - 1)
    else:
        length_out = int(length_out)
        by = width / (length_out - 1) 
    out = [float(start)]*length_out
    for i in range(1, length_out):
        out[i] += by * i
    if abs(start + by * length_out - end) < eps:
        out.append(end)
    return out
# seq(0,10,1)

# ADP for CSOC
# 
# Created by Ali S. Mazloom on 11/11/19.
# Copyright © 2019 Ali S. Mazloom. All rights reserved.

import numpy as np
import random
import time
import csv
import sys
import os
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

def main():
  os.system('clear')

# initialization
  v_bar = np.zeros((336,200,8,8)).astype(float)
  v_hat = 0
  learning_rate = 1
  # discount factor
  beta = 0.9
  # exploration parameter - percentage of iterations for exploration phase
  p = 0.1
  #total number of iterations
  N = 1000
  # number of iterations in exploration phase
  M = p * N
  # regular number of sensor and analysts
  max_sensors = 10
  regular_analysts = 10

  # alert arrival rate per sensor per hour 
  #aar = 50+0.892857
  aar = 50
  #service_rate_per_analyst_per_hour 
  #srp= 51+1.0625
  srp = 51
  # service rate
  rho = aar / srp
  totalTime = 14 * 24
  w1 = 0.7
  w2 = 1 - w1
  # additional events
  ae = 6
  # calculate gradient and mean square error 
  grad = 0
  MSE = [0] * N
  # queue_l = (rho ** 2)/(2 * (1 - rho))
  # record every component value for troubleshooting
  report = np.zeros((totalTime + 1,16))
  # # average service time for all analysts
  # AvgT = (1 / srp) + rho / (2 * srp * (1 - rho))
  #nn=[0] * N
  # number of sensors reportorting alerts at time t
  def generate_active_sensor(max_sensors):
    x = np.random.randint(0 , max_sensors)
    x = 10
    return x

  # number of alerts arriving at time t; conforms to poisson dist.
  def generate_alerts(lam , sensors):
    ga = np.random.poisson(lam , sensors)
    return ga

  def service_alerts(alerts , extra_analysts):
    # total analysts
    a = regular_analysts + extra_analysts
    b = int(0)
    m = srp * a
    #a = alerts // srp
    if (alerts >= m):
      b = (alerts % srp) 
    else:
      b = 0
    return b

  # add stochastic exogenous demand to the process
  def generate_additional_alerts(addLam , number_of_additional_events):
    gaa = np.random.poisson(addLam , number_of_additional_events)
    timeIndex = np.random.randint(1 , totalTime , size = number_of_additional_events)
    #gaa = [33, 47, 70, 69, 51, 47]
    #timeIndex = [32, 54, 80, 170, 210, 320]
    #print(gaa,timeIndex)
    return timeIndex , gaa

  # def normalizeBack(st1):
  #   max_queue = 50
  #   if (st1 > max_queue):
  #     nst1 = np.absolute((st1 / max_queue) - 1)
  #   else:
  #     nst1 = np.absolute(1 - (st1 / max_queue))
  #   return nst1
  def normalizeBack(st1):
    if (st1 > 70):
      nst1 = 0
    else:
      nst1 = np.absolute((st1 / 70) - 1)
    return nst1

  def normalizeRes(st2):
    nst2 = 10 * st2
    if (nst2 > 1):
      nst2 = 1 
    #print('iter: '+str(j)+' t: '+str(t))
    #print((round(float(nst2),3)))
    return nst2

  def reward(s1,s2,x,time):
    r = 0
    if (totalTime != time):
      r = w1 * normalizeBack(s1) + w2 * normalizeRes((s2 - x) / (totalTime - time))
      #print('rewfunc'+str(((s2 - x) / (totalTime - time))))
    return r

  def alpha_decay(iter,alpha):
    #for 10000 iterations
    #bigM = 50000000000 * 7
    #for 1000 iterations
    bigM = 50000000 * 7
    #for 100 iterations
    #bigM = 50000 * 7
    u = iter ** 2 / (bigM + iter)
    alpha = alpha / (1 + u)
    return alpha

  for j in range(1 , N):
    pds = np.zeros((2,totalTime))
    s = np.zeros((2,totalTime+1))
    grad = 0
    # reserved resource
    rsrvd = 7
    additionalTime = [0] * ae
    additionalAlert = [0] * ae
    additionalTime , additionalAlert = generate_additional_alerts(aar , ae)
    #print('extra alerts time' + str(additionalTime))
    #print('extra alerts     ' + str(additionalAlert))
    x = 0
    # generate regular demand for time t
    ac = generate_active_sensor(max_sensors)
    z_generated = np.sum(generate_alerts(aar , ac))
    s[1,0] = rsrvd
    for t in range(totalTime):
      x = 0
      report[t,0] = j
      report[t,1] = t
      if (t==0):
        generated = z_generated
        report[0,2] = z_generated
        report[0,13] = z_generated
      report[t,3] = s[0,t]
      report[t,4] = s[1,t]
      # if (t==0):
      #   report[0,13] = z_generated 
      # if (j<2):
      #   j = N-1
      if (j <= M):
        x_hat = (generated + s[0,t]) / srp
        if (x_hat > regular_analysts):
          x = np.random.poisson(np.ceil(x_hat - regular_analysts) , 1)
          if (rsrvd < x):
            x = 0
        else:
          x = 0
        if (rsrvd > 0):
          rsrvd -= x
        else:
          rsrvd = 0
        c = reward(s[0,t] , s[1,t] , x , t)
        report[t,5] = x
        report[t,6] = c      
        v_hat_explore = c + beta * v_bar[t , int(pds[0,t]) , int(pds[1,t]) , x]
        v_hat = v_hat_explore
        # introducing post-decision state variable at time t
        pds[0,t] = service_alerts(generated + s[0,t] , x)
        pds[1,t] = rsrvd
      else:
        v_hat_exploit = [0] * 8
        for x in range(rsrvd + 1):
          c = reward(s[0,t] , s[1,t] , x , t)
          v_hat_exploit[x] = c + beta * v_bar[t , int(pds[0,t]) , int(pds[1,t]) , x]
          #print(x , s[0,t] , s[1,t] , c , pds[0,t] , pds[1,t] , v_hat_exploit[x])
        temp_max = 0
        temp_index = 0
        for i in range(rsrvd + 1):
          if (v_hat_exploit[i] > temp_max):
            temp_max = v_hat_exploit[i]
            temp_index = i
        v_hat = v_hat_exploit[temp_index]
        x = temp_index
        # print('\n' + str(v_hat) +' x: ' + str(x))
        if (rsrvd > 0):
          rsrvd -= x
        # introducing post-decision state variable at time t
        pds[0,t] = service_alerts(generated + s[0,t] , x)
        pds[1,t] = rsrvd
        # if (rsrvd < 0):
        #   rsrvd = 0
        #   x = 0
        report[t,5] = x
        report[t,6] = reward(s[0,t] , s[1,t], x , t)
      report[t,7] = v_hat
      grad += (v_hat - v_bar[t , int(pds[0,t]) , int(pds[1,t]) , x]) ** 2
      v_bar[t , int(pds[0,t]) , int(pds[1,t]) , x] = (1 - learning_rate) * v_bar[t , int(pds[0,t]) , int(pds[1,t]) , x] + learning_rate * v_hat
      if (x != 0):
        print(t,pds[0,t],pds[1,t],x,v_bar[t , int(pds[0,t]) , int(pds[1,t]) , x])
      report[t,8] = v_bar[t , int(pds[0,t]) , int(pds[1,t]) , x]
      report[t,14] = grad
      # generate alerts for time t+1
      generated = np.sum(generate_alerts(aar , ac))
      report[t+1,2] = generated
      # add exogenous demand to regular alert arrivals
      for i in range(ae):
        if (t == additionalTime[i]):
          generated += additionalAlert[i]
      report[t+1,13] = generated
      # calculate pre-decision state for time t+1
      if (t < totalTime):
        s[0,t+1] = pds[0,t]
        s[1,t+1] = rsrvd
      report[t,9] = pds[0,t]
      report[t,10] = pds[1,t]
      report[t,11] = s[0,t+1]
      report[t,12] = s[1,t+1]
      report[t,15] = report[t,13] - report[t,2]
    # get highest queue length
    #nn[j] = np.max(report[:,11])
    #report[0,16] = learning_rate
    
    learning_rate = alpha_decay(j , learning_rate)
    #print(' vhat='+str(round(v_hat,3))+' vbar='+str(round(v_bar,3)))
    MSE[j] = (MSE[j-1] + grad) / j
    print('iter: ' + str(j) + '\nmse:' + str(round(float(MSE[j]),10)))
    print('alpha: ' + str(learning_rate))
    #print(learning_rate)
  #print(np.max(nn))
  # output system variables to dataset
  filename = 'TestsRL2.csv' 
  np.savetxt(filename, report, delimiter=",")
  plt.plot(MSE)
  plt.ylabel('Mean Square Error')
  plt.show()
  plt.plot(report[:,11])
  plt.ylabel('Queue Length')
  plt.show()
  #print(v_bar[0,0,7,0])
  # for t in range(v_bar.shape[0]):
  #   for s1 in range(v_bar.shape[1]):
  #      for s2 in range(v_bar.shape[2]):
  #        for x in range(8):

if __name__ == '__main__':
  main()

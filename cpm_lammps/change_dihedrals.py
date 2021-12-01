import numpy as np
import math
import matplotlib.pyplot as plt

import time
start = time.time()


#file_angles = 'ellipsoid.txt'
file_dihedrals = 'azdh.txt'

#data1 = np.loadtxt(file_angles, skiprows=4)
#index = data1[:,0];
#cosine2 = data1[:,1];

data2 = np.loadtxt(file_dihedrals, skiprows=0)
dindex = data2[:,0];
d1 = data2[:,2];
d2 = data2[:,3];
d3 = data2[:,4];
d4 = data2[:,5];

dindex = list(map(int, dindex))
d1 = list(map(int, d1))
d2 = list(map(int, d2))
d3 = list(map(int, d3))
d4 = list(map(int, d4))

dindex = np.array(dindex)
d1 = np.array(d1)
d2 = np.array(d2)
d3 = np.array(d3)
d4 = np.array(d4)

# draw random number from some linear distribution between 0 and 1
# go through all entries and determine if switching will happen
switch=[]
switchback=[]
tc=[]
ct=[]
angles_switched=[]
angles_nonswitched=[]
#for x in cosine2:
#for x in dindex:

NAZO = len(dindex)

# read in or keep which azo is currently trans or cis

try:
    f = open("azid_trans.txt", "r")
    print("TRANS EXISTS")
    azid_trans = np.loadtxt("azid_trans.txt", skiprows=0, usecols=(1)).astype(int)
except IOError:    #This means that the file does not exist (or some other IOError)
    print("Oops, no TRANS file by that name")
    azid_trans = np.ones((NAZO,), dtype=np.int16)


try:
    f = open("azid_cis.txt", "r")
    print("CIS EXISTS")
    azid_cis = np.loadtxt("azid_cis.txt", skiprows=0, usecols=(1)).astype(int)
except IOError:    #This means that the file does not exist (or some other IOError)
    print("Oops, no CIS file by that name")
    azid_cis = np.zeros((NAZO,), dtype=np.int16)


#azid_trans = np.ones((NAZO,), dtype=np.int16)
#azid_cis = np.zeros((NAZO,), dtype=np.int16)

tc_ids = np.zeros((NAZO,), dtype=np.int16)
ct_ids = np.zeros((NAZO,), dtype=np.int16)



# DETERMINE SWITCHES FROM TRANS TO CIS
for ii, x in enumerate(dindex):

    # draw random number
    r = np.random.random_sample()
    if (r < 0.0188 and azid_trans[ii] == 1):
        switch.append(1)
        tc.append(23)
        #ct.append(24)
        #angles_switched.append(np.arccos(np.sqrt(x))*180.0/np.pi)
        
        azid_trans[ii] = 0
        azid_cis[ii] = 1

        tc_ids[ii] = 1

    else:
        switch.append(0)
        tc.append(19)
        #ct.append(19)
        #angles_nonswitched.append(np.arccos(np.sqrt(x))*180.0/np.pi)


# DETERMINE SWITCHES FROM CIS TO TRANS
for ii, x in enumerate(dindex):

    # draw random number
    r = np.random.random_sample()
    if (r < 0.0012 and azid_cis[ii] == 1):
        switchback.append(1)
        #tc.append(23)
        ct.append(24)
        #angles_switched.append(np.arccos(np.sqrt(x))*180.0/np.pi)

        azid_trans[ii] = 1
        azid_cis[ii] = 0

        ct_ids[ii] = 1  

    else:
        switchback.append(0)
        #tc.append(19)
        ct.append(19)
        #angles_nonswitched.append(np.arccos(np.sqrt(x))*180.0/np.pi)

azdh_tc = list(zip(dindex, tc, d1, d2, d3, d4))
azdh_ct = list(zip(dindex, ct, d1, d2, d3, d4))
#azdh_tc = np.vstack((dindex, tc, d1, d2, d3, d4))
#azdh_ct = np.vstack((dindex, ct, d1, d2, d3, d4))
#print(list(azdh_tc))
#exit()

Nswitch=sum(switch)
Nazos=len(switch)

Nswitchback=sum(switchback)
Nazos_=len(switchback)

#switched_avg_angle=np.mean(angles_switched)
#nonswitched_avg_angle=np.mean(angles_nonswitched)
#switched_std_angle=np.std(angles_switched)
#nonswitched_std_angle=np.std(angles_nonswitched)

### append this to respective files
 
with open("Nswitch.txt", "a") as myfile:
    myfile.write("%s %s %s\n" % (int(Nazos), int(Nswitch), 1.0*Nswitch/Nazos))

with open("Nswitchback.txt", "a") as myfile:
    myfile.write("%s %s %s\n" % (int(Nazos_), int(Nswitchback), 1.0*Nswitchback/Nazos_))


with open("azid_trans.txt" , "w") as myfile:
    for jj, item in enumerate(azid_trans):
        myfile.write("%s %s\n" % (jj, item))

with open("azid_cis.txt", "w") as myfile:
    for jj, item in enumerate(azid_cis):
        myfile.write("%s %s\n" % (jj, item))

with open("tc_ids.txt" , "w") as myfile:
    for jj, item in enumerate(tc_ids):
        myfile.write("%s %s\n" % (jj, item))

with open("ct_ids.txt", "w") as myfile:
    for jj, item in enumerate(ct_ids):
        myfile.write("%s %s\n" % (jj, item))

#with open("avg_angles.txt", "a") as myfile2:
#    myfile2.write("%s %s %s %s\n" % (switched_avg_angle, switched_std_angle, nonswitched_avg_angle, nonswitched_std_angle))

#print(azdh_tc)
#print(list(azdh_tc))


ftc = open('azdh_tc.txt', 'w')
for item in azdh_tc:
    ftc.write("%s %s %s %s %s %s\n" % (item[0], item[1], item[2], item[3], item[4], item[5]))
ftc.close()

fct = open('azdh_ct.txt', 'w')
for item in azdh_ct:
    fct.write("%s %s %s %s %s %s\n" % (item[0], item[1], item[2], item[3], item[4], item[5]))
fct.close()


end = time.time()
#print(end - start)

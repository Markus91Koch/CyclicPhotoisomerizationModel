import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import colors
import matplotlib as mpl
import os
import glob


#myfiles = glob.glob("ev_*.txt")
#myfiles = ["ev_0002.txt", "ev_0003.txt", "ev_0004.txt","ev_0005.txt", "ev_0006.txt","ev_0007.txt","ev_0008.txt","ev_0009.txt","ev_0010.txt",]
#myfiles = ["ev_0043.txt"]
#myfiles.sort()
#n = len(myfiles)
#Nmol = 36
#Nazo = 3*Nmol
#mol_indices = np.zeros((n, Nmol))
#azo_indices = np.zeros((n,Nmol,3))


tc_mat_ = np.loadtxt("tc_CONT.txt")
ct_mat_ = np.loadtxt("ct_CONT.txt")


#print(trans_mat.shape)



#### PREPEND "ONES" at first szep!

tc_mat = np.swapaxes(tc_mat_,0,1)
ct_mat = np.swapaxes(ct_mat_,0,1)

#print(tc_mat.shape)

n = tc_mat.shape[0]
Nazo = tc_mat.shape[1]
Nmol = int(Nazo/3)

#trans_mat_old = np.copy(trans_mat)
#trans_mat = np.ones((n+1, Nazo))
#trans_mat[1:,:] = trans_mat_old


#n = trans_mat.shape[0]
#Nazo = trans_mat.shape[1]
#Nmol = int(Nazo/3)


print(n, Nazo, Nmol)


print(tc_mat.shape)
print(ct_mat.shape)


#trans_time = np.zeros((n, Nazo))
cis_time = np.zeros((n,Nazo))
now_cis =  np.zeros(Nazo)
longest_times=[]

trans_mat = np.ones((n,Nazo))
isomer = np.zeros((n,4))



for i in range(n):
    for j in range(Nazo):
        if tc_mat[i,j] == 1:

            now_cis[j] = 0.5 # cis time will be at least for half a cycle
            cis_time[i,j] = now_cis[j]

            trans_mat[i,j] = 0

            print(i,j, tc_mat[i,j], ct_mat[i,j], trans_mat[i,j], cis_time[i,j], now_cis[j])
 

            if (ct_mat[i,j] == 1):

                longest_times.append(now_cis[j])
                now_cis[j] = 0

                print(i,j,tc_mat[i,j], ct_mat[i,j], trans_mat[i,j], cis_time[i,j], now_cis[j])
                

        elif ((ct_mat[i,j] == 0) and (now_cis[j] != 0)):

            now_cis[j] += 1
            cis_time[i,j] = now_cis[j]

            trans_mat[i,j] = 0

            print(i,j, tc_mat[i,j], ct_mat[i,j], trans_mat[i,j], cis_time[i,j], now_cis[j])

        elif ((ct_mat[i,j] == 1) and (now_cis[j] != 0)):

            now_cis[j] += 1
            cis_time[i,j] = now_cis[j]
            longest_times.append(now_cis[j])
            now_cis[j] = 0

            trans_mat[i,j] = 0

            print(i,j,tc_mat[i,j], ct_mat[i,j], trans_mat[i,j], cis_time[i,j], now_cis[j])
       



#exit()


trans_mat_old = np.copy(trans_mat)
trans_mat = np.ones((n+1, Nazo))
trans_mat[1:,:] = trans_mat_old


N = trans_mat.shape[0]
#Nazo = trans_mat.shape[1]
#Nmol = int(Nazo/3)


trimer_mat = np.zeros((N, Nmol))
for i in range(N):
    for j in range(Nmol):
        cumul = np.sum(trans_mat[i,j*3:j*3+3])
        trimer_mat[i,j] = cumul


trans_hist = np.zeros((N,4))
for i in range(N):
    trans_hist[i,0] = (trimer_mat[i,:] == 0).sum()
    trans_hist[i,1] = (trimer_mat[i,:] == 1).sum()
    trans_hist[i,2] = (trimer_mat[i,:] == 2).sum()
    trans_hist[i,3] = (trimer_mat[i,:] == 3).sum()



T = np.arange(N)


#trans_tot_time=[]
#cis_tot_time=[]
#for i in range(N)
#    trans=np.sum(trans_mat[i,:])
trans_tot_time = np.sum(trans_mat, axis=1)
cis_tot_time = np.ones(N)*Nazo - trans_tot_time

#print(trans_tot_time)
#print(cis_tot_time)


for j in range(Nazo):
    print(np.sum(cis_time[:,j]))


#tc_mol_mat
tc_mol_mat = np.zeros((n,Nmol))
ct_mol_mat = np.zeros((n,Nmol))
for i in range(n):
    for j in range(Nmol):
        tc_mol_mat[i,j] = np.sum(tc_mat[i,j*3:j*3+1])
        ct_mol_mat[i,j] = np.sum(ct_mat[i,j*3:j*3+1])



tc_events_azo = np.sum(tc_mat, axis=0)
tc_events_time = np.sum(tc_mat, axis=1)

ct_events_azo = np.sum(ct_mat, axis=0)
ct_events_time = np.sum(ct_mat, axis=1)

tc_events_mol = np.sum(tc_mol_mat, axis=0)
ct_events_mol = np.sum(ct_mol_mat, axis=0)




tc_nbins = int(np.ceil(np.max(tc_events_azo))+1)
tc_mybins = np.linspace(0,tc_nbins, tc_nbins+1)
tc_mybincenter = np.array(tc_mybins[1:]) - 0.5
tc_hist, tc_edges =  np.histogram(tc_events_azo, tc_mybins)

ct_nbins = int(np.ceil(np.max(ct_events_azo))+1)
ct_mybins = np.linspace(0,ct_nbins, ct_nbins+1)
ct_mybincenter = np.array(ct_mybins[1:]) - 0.5
ct_hist, ct_edges =  np.histogram(ct_events_azo, ct_mybins)



TC_nbins = int(np.ceil(np.max(tc_events_mol))+1)
TC_mybins = np.linspace(0,TC_nbins, TC_nbins+1)
TC_mybincenter = np.array(TC_mybins[1:]) - 0.5
TC_hist, TC_edges =  np.histogram(tc_events_mol, TC_mybins)

CT_nbins = int(np.ceil(np.max(ct_events_mol))+1)
CT_mybins = np.linspace(0,CT_nbins, CT_nbins+1)
CT_mybincenter = np.array(CT_mybins[1:]) - 0.5
CT_hist, CT_edges =  np.histogram(ct_events_mol, CT_mybins)

#print(tc_events_mol)
#print(tc_events_time)



#exit()




#print(cis_time)
#print(longest_times)

print(np.max(longest_times))
nbins =   int(np.ceil(np.max(longest_times))+1)







mybins = np.linspace(0,nbins,nbins+1)
mybincenter = np.array(mybins[1:])-0.5


#print(nbins)
#print(mybins)
#print(mybincenter)

hist, bin_edges = np.histogram(longest_times, mybins)
#print(hist.shape)
#print(bin_edges.shape)
#print(bin_edges)

print("LONGEST TIME IN CIS")
print(np.max(longest_times))

print("\nSHORTEST TIME IN CIS")
print(np.min(longest_times))

print("\nAVRAGE TIME IN CIS")
print(np.mean(longest_times))


# get the following statistics:
# - steady state count (or percentage) of ttt,ttc,tcc,ccc
# - max, min (range) and average time in cis
# - average number of tc / ct events of one arm
# - average number of tc  /ct events in a molecule
# - min / max number of detected tc/ct events in arms / molecules
# - what is how distributed? are tc/ct events poisson distr?
# - are cis times poisson distr?
# - make a fit to estimate how long it takes for steady state to be reached??
# do all this with actual pico or nanoseconds .. 
# are these stats realistic? how about correlations between events?

# biased simulations? overall low chance of trans-cis event but chance is increased if a cis group exists in the neighboring molecule

fig001, ax001 = plt.subplots()
cmap = plt.cm.get_cmap("gnuplot2_r", 2)
im = ax001.imshow(trans_mat, cmap = cmap)
cbar = fig001.colorbar(im)
ax001.set_xlabel("azo index")
ax001.set_ylabel("no. cycle (time)")
plt.savefig("trans_matrix.png", dpi=300)
plt.savefig("trans_matrix.pdf", dpi=300)

fig002, ax002 = plt.subplots()
cmap = plt.cm.get_cmap("gnuplot2_r", 4)
im = ax002.imshow(trimer_mat, vmin=0, vmax=3, cmap=cmap)
cbar = fig002.colorbar(im)
ax002.set_xlabel("molecule index")
ax002.set_ylabel("no. cycle (time)")
plt.savefig("trimer_matrix.png", dpi=300)
plt.savefig("trimer_matrix.pdf", dpi=300)


fig003, ax003 = plt.subplots()
ax003.plot(T, trans_hist[:,0]/Nmol, c="xkcd:blue" , label="ccc")
ax003.plot(T, trans_hist[:,1]/Nmol, c="xkcd:red" , label="tcc")
ax003.plot(T, trans_hist[:,2]/Nmol, c="xkcd:green" , label="ttc")
ax003.plot(T, trans_hist[:,3]/Nmol, c="xkcd:black" , label="ttt")
ax003.legend()
ax003.set_xlabel("no. cycle (time)")
ax003.set_ylabel("count of TrisAzo isomers")
plt.savefig("isomer_count_vs_time.png", dpi=300)
plt.savefig("isomer_count_vs_time.pdf", dpi=300)

np.savetxt("isomer_count_vs_time.txt", np.c_[T, trans_hist[:,0]/Nmol, trans_hist[:,1]/Nmol, trans_hist[:,2]/Nmol, trans_hist[:,3]/Nmol], header="time ccc tcc ttc ttt")
np.savetxt("isomer_count_total_vs_time.txt", np.c_[T, trans_hist[:,0], trans_hist[:,1], trans_hist[:,2], trans_hist[:,3]], header="time ccc tcc ttc ttt")



#plt.show()


fig0044, ax0044 = plt.subplots()
ax0044.plot(T, trans_tot_time/Nazo, c="xkcd:fuchsia" , label="trans")
ax0044.plot(T, cis_tot_time/Nazo, c="xkcd:electric blue" , label="cis")
ax0044.legend()
ax0044.set_xlabel("no. cycle (time)")
ax0044.set_ylabel("count of (azo arm) isomers")
plt.savefig("trans_cis_count_vs_time.png", dpi=300)
plt.savefig("trans_cis_count_vs_time.pdf", dpi=300)

np.savetxt("trans_cis_count_vs_time.txt", np.c_[T, trans_tot_time/Nazo, cis_tot_time/Nazo, trans_tot_time, cis_tot_time], header="time trans_rel cis_rel  trans_abs cis_abs")




fig00, ax00 = plt.subplots()
cmap = plt.cm.get_cmap("gnuplot2_r", 2)
im = ax00.imshow(tc_mat, cmap = cmap)
cbar = fig00.colorbar(im)
ax00.set_xlabel("no. cycle (time)")
ax00.set_ylabel("count of trans-cis events")
plt.savefig("tc_events_vs_time.png", dpi=300)
plt.savefig("tc_events_vs_time.pdf", dpi=300)



fig000, ax000 = plt.subplots()
cmap = plt.cm.get_cmap("gnuplot2_r", 2)
im = ax000.imshow(ct_mat, cmap = cmap)
cbar = fig000.colorbar(im)
ax000.set_xlabel("no. cycle (time)")
ax000.set_ylabel("count of cis-trans events")
plt.savefig("ct_events_vs_time.png", dpi=300)
plt.savefig("ct_events_vs_time.pdf", dpi=300)




fig007, ax007 = plt.subplots()
cmap = plt.cm.get_cmap("gnuplot2_r", 2)
im = ax007.imshow(tc_mol_mat, cmap = cmap)
cbar = fig007.colorbar(im)
ax007.set_xlabel("no. cycle (time)")
ax007.set_ylabel("count of trans-cis events in a trimer")
plt.savefig("tc_events_vs_time.png", dpi=300)
plt.savefig("tc_events_vs_time.pdf", dpi=300)



fig008, ax008 = plt.subplots()
cmap = plt.cm.get_cmap("gnuplot2_r", 2)
im = ax008.imshow(ct_mol_mat, cmap = cmap)
cbar = fig008.colorbar(im)
ax008.set_xlabel("no. cycle (time)")
ax008.set_ylabel("count of cis-trans events in a trimer")
plt.savefig("ct_events_vs_time.png", dpi=300)
plt.savefig("ct_events_vs_time.pdf", dpi=300)




fig0, ax0 = plt.subplots()
cmap = plt.cm.get_cmap("gnuplot2_r", np.max(cis_time)+1)
im = ax0.imshow(cis_time, cmap = cmap)
cbar = fig0.colorbar(im)
ax00.set_xlabel("no. cycle (time)")
ax00.set_ylabel("cumulative # of cycles in cis")
plt.savefig("cis_time_cumul_vs_time.png", dpi=300)
plt.savefig("cis_time_cumul_vs_time.pdf", dpi=300)
# CIS TIMES THAT DID NOT END WITH A CT-EVENT ARE NOT INCLUDED (i.e. if the arm is still in cis at the end of the simulation these times are not included!!)


fig1, ax1 = plt.subplots()
ax1.bar(mybincenter, hist)
ax1.set_xlabel("time period in cis state")
ax1.set_ylabel("count")
plt.savefig("cis_time_hist.png", dpi=300)
plt.savefig("cis_time_hist.pdf", dpi=300)


fig2, ax2 = plt.subplots()
ax2.bar(tc_mybincenter, tc_hist)
ax2.set_xlabel("number of trans-cis events in one arm")
ax2.set_ylabel("count")
plt.savefig("tc_per_arm_hist.png", dpi=300)
plt.savefig("tc_per_arm_hist.pdf", dpi=300)
# ALSO DO THIS HIST PER MOLECULE _ HOW MANY EVENTS ARE IN ONE MOLECULE OVER THE COURSE OF THE SIMULATION


fig3, ax3 = plt.subplots()
ax3.bar(ct_mybincenter, ct_hist)
ax3.set_xlabel("number of cis-trans events in one arm")
ax3.set_ylabel("count")
plt.savefig("ct_per_arm_hist.png", dpi=300)
plt.savefig("ct_per_arm_hist.pdf", dpi=300)


fig20, ax20 = plt.subplots()
ax20.bar(TC_mybincenter, TC_hist)
ax20.set_xlabel("number of trans-cis events in one Trimer")
ax20.set_ylabel("count")
plt.savefig("tc_per_mol_hist.png", dpi=300)
plt.savefig("tc_per_mol_hist.pdf", dpi=300)
# ALSO DO THIS HIST PER MOLECULE _ HOW MANY EVENTS ARE IN ONE MOLECULE OVER THE COURSE OF THE SIMULATION


fig30, ax30 = plt.subplots()
ax30.bar(CT_mybincenter, CT_hist)
ax30.set_xlabel("number of cis-trans events in one Trimer")
ax30.set_ylabel("count")
plt.savefig("ct_per_mol_hist.png", dpi=300)
plt.savefig("ct_per_mol_hist.pdf", dpi=300)




t = np.arange(1,n+1)
fig4, ax4 = plt.subplots()
ax4.plot(t,tc_events_time, "xkcd:pink", label="trans-cis")
ax4.plot(t,ct_events_time, "xkcd:electric blue", label="cis-trans")
ax4.legend()
ax4.set_ylabel("number of cycles (time)")
ax4.set_ylabel("number of switching events")
plt.savefig("tc-ct_events_vs_time.png", dpi=300)
plt.savefig("tc-ct_events_vs_time.pdf", dpi=300)





















exit()
# create matrix that contains which molecules contain 0,1,2 or 3 trans groups

trimer_mat = np.zeros((n, Nmol))
print(trimer_mat.shape)


#trimer_mat[0,:] = 1


for i in range(n):
    for j in range(Nmol):
        cumul = np.sum(trans_mat[i,j*3:j*3+3])
        #print(trans_mat[i,j*3:j*3+3])
        trimer_mat[i,j] = cumul
        #exit()
        #trimer_mat.append(cumul)


#trimer_mat = np.convolve(trans_mat, np.ones(3, dtype=np.int), mode='valid')

print(trimer_mat[0,:])
print((trimer_mat[0,:] == 3).sum())
#print(np.bincount(trimer_mat[i,:])


#exit()
trans_hist = np.zeros((n,4))



for i in range(n):
    trans_hist[i,0] = (trimer_mat[i,:] == 0).sum()
    trans_hist[i,1] = (trimer_mat[i,:] == 1).sum()
    trans_hist[i,2] = (trimer_mat[i,:] == 2).sum()
    trans_hist[i,3] = (trimer_mat[i,:] == 3).sum()



#print(trimer_mat[:,0])
t = np.arange(n)
#exit()

fig, ax = plt.subplots()
#cmap = colors.ListedColormap(["white","blue"])
#cmap = plt.cm.viridis
cmap = plt.cm.get_cmap("gnuplot2_r", 2)
#bounds = np.linspace(0,1,2)#[0,.5,1]
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
im = ax.imshow(trans_mat, vmin=0, vmax=1, cmap = cmap)
cbar = fig.colorbar(im)
#plt.savefig()
#plt.show()

fig1, ax1 = plt.subplots()

cmap = plt.cm.get_cmap("gnuplot2_r", 4)
im = ax1.imshow(trimer_mat, vmin=0, vmax=3, cmap=cmap)
cbar = fig1.colorbar(im)
#plt.show(t, trimer_mat[:,0], c="xkcd:blac" , label="0 trans")
#plt.show(t, trimer_mat[:,1], c="xkcd:dark purple", label="1 trans")
#plt.show(t, trimer_mat[:,2], c="xkcd:purple", label="2 trans")
#plt.show(t, trimer_mat[:,3], c="xkcd:pink", label="3 trans")
#plt.show()




fig2, ax2 = plt.subplots()
ax2.plot(t, trans_hist[:,0]/Nmol, c="xkcd:fuchsia" , label="ccc")
ax2.plot(t, trans_hist[:,1]/Nmol, c="xkcd:red orange" , label="tcc")
ax2.plot(t, trans_hist[:,2]/Nmol, c="xkcd:robins egg" , label="ttc")
ax2.plot(t, trans_hist[:,3]/Nmol, c="xkcd:electric blue" , label="ttt")


ax2.legend()

plt.show()



# durcschnittliche zeit in cis, max und min zeit in cis berechnen!!!!



exit()



for i, f in enumerate(myfiles):


    print(i)
    idx1 = np.loadtxt(f, usecols=(2))
    print(idx1)
    

    #if np.isscalar(idx1) == False:
    #    print("YO")
    idx1 = np.atleast_1d(idx1)
    
    #print(idx1)
    #exit()

    mol_idx = np.floor(idx1/114)
    print(mol_idx)
    
    buff_idx = idx1 - 114*mol_idx
    print(buff_idx)

    # alternating!!!!
    azo_idx = [0 if x<51 else 1 if x==51 else 2 for x in buff_idx]
    print(azo_idx)

    # how to treat single values or empty files
        

    # how to store that multiple switches occur in one molecule?
    for j, midx in enumerate(mol_idx):
          
        mol_indices[i,int(midx)] = mol_indices[i,int(midx)] + 1
   
    for j, aidx in enumerate(azo_idx):
        azo_indices[i, int(mol_idx[j]), int(aidx)] += 1


    #exit()

print(mol_indices)
#print(mol_indices.shape)

fig, ax = plt.subplots()
im = ax.imshow(mol_indices)
plt.show()

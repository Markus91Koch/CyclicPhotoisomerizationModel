import numpy as np
import math
import matplotlib.pyplot as plt
#plt.switch_backend('Agg')
#plt.switch_backend('Qt4Agg')
import csv
from itertools import chain
from itertools import compress
import os
import pandas as pd
import seaborn as sns
#print os.getcwd()
import scipy.special

def rm1(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def rm2(x,N):
    return np.convolve(x, np.ones(N)/N, mode='valid')


def rm3(x,N):
    XS = pd.Series(x)
    return np.r_[x[0],XS.rolling(window=N).mean().dropna(), x[-1]]

def rm3e(x,N):
    XS = pd.Series(x)
    return np.r_[0.,XS.rolling(window=N).std().dropna(), 0.]


def isomer_predict(t_arms, p_tc, p_ct):
    t_arms = int(t_arms)
    return scipy.special.comb(3, t_arms, exact=True) * (p_ct**t_arms * p_tc**(3-t_arms)) / (p_tc + p_ct)**3

arbeitspfad=os.getcwd()

#os.chdir("../Dreiding/new_structures/")
os.chdir("../")

#arbeitspfad='/home/koch_local/taurus/lmpruns/xclusters/new_ccc'
#tarbeitspfad='/home/koch_local/taurus/lmpruns/xclusters/smaller'

#cluster_type=["line", "alt", "chiral"]
#mylabel=["linear", "alt", "twist"]
#cluster_type=["1percent", "5percent", "10percent", "BOTH_01_01", "BOTH_05_05", "BOTH_1_1", "BOTH_2_2", "BOTH_5_5"]
#mylabel=[r"1%", r"5%", r"10%", r"0.1%-0.1%" ,  r"0.5%-0.5%" ,   r"1%-1%", r"2%-2%"  , r"5%-5%"]
#cluster_type=["BOTH_01_01", "BOTH_05_05","BOTH_1_1", "BOTH_2_2", "BOTH_5_5"]
#mylabel=[r"0.1%-0.1%" ,  r"0.5%-0.5%" ,r"1%-1%", r"2%-2%"  , r"5%-5%"]

cluster_type=["BOTH_0188_0012"]
mylabel=[r"1.88%-0.12%"]

#cluster_type=["BOTH_two_one_third"]
#mylabel=[r"1.33%-0.67%"]


#style=[":", "-.", "--", "-"]
style=["-","-","-", "--", "--" ,"--", "--", "--"]

#mylabel=[r"$E_\mathrm{bind}$", r"$E_\mathrm{Coul}$",r"$E_\mathrm{vdW}$",r"$E_\mathrm{hb}$"]
#mylabel=["total", "Coul", "LJ", "HB"]

#N=[6, 12, 18 , 36]
#folders=["6", "12", "18", "36_inf"]
#subfolder="eq_10/rerun_chain"

#N=[2,3, 6, 12, 18, 36]
#folders=["2", "3", "6", "12", "18", "36"]
#folders=[""]

N_values = [36]
#folders = ["stride10", "cut_stride10", "stride10", "stride10", "stride10", "stride10", "stride10", "stride10"]



#N=[36]
#folders=["36"]
#subfolder="run_25/rerun_chain"
subfolder=""



length='npairs.txt'
#file1='distances_com.txt'
#file2='distances_com_mol.txt'

#file1="interactions_mol.txt"
#file1="spatial_corr_func_avg.txt"
#file2="spatial_corr_func_avg_center.txt"

#file1="orientation_avg_center.txt"
#file1="trans_cis_count_vs_time.txt"
file1="isomer_count_vs_time.txt"
#file1='orientation_fullavg_com.txt'
#file2='orientation_com_mol.txt'

file1center='distances_center.txt'
file2center='distances_center_mol.txt'
file1bigcenter='distances_bigcenter.txt'
file2bigcenter='distances_bigcenter_mol.txt'

Efile1 = "E_inter.txt"
Efile2 = "E_inter_mol.txt"

hbfile1 = "hb_mean.txt"

#palette = ["xkcd:dark pink",
#"xkcd:rosa",
#"xkcd:electric blue",
#"xkcd:neon blue",
#"xkcd:ocean green",
#"xkcd:lightish green",
#"xkcd:tangerine",
#"xkcd:apricot"]

#palette = ["xkcd:electric blue", 
#"xkcd:orange red",
#"xkcd:aquamarine",
#"xkcd:goldenrod"
#]


palette = ["xkcd:electric blue", 
"xkcd:aquamarine",
#"xkcd:orange red",
"xkcd:goldenrod",
"xkcd:electric blue",
"xkcd:orange red",
"xkcd:aquamarine"
]

palette = ["xkcd:electric blue",
"xkcd:aquamarine",
#"xkcd:orange red",
"xkcd:goldenrod",
"xkcd:violet",
"xkcd:magenta",
"xkcd:electric blue",
"xkcd:orange red",
"xkcd:aquamarine"
]

palette = [#"xkcd:electric blue",
#"xkcd:aquamarine",
#"xkcd:orange red",
#"xkcd:goldenrod",
#"xkcd:neon purple",
#"xkcd:vibrant green",
"xkcd:electric blue",
"xkcd:aquamarine",
"xkcd:goldenrod",
"xkcd:orange red"
]



clustersize=[]
Nmol=[]
Npairs=[]


Nmol1=[]
Npairs1=[]


o_avg=[]
o_std=[]

E_avg=[]
E_std=[]

corr_avg = []
corr_std = []
corr_lo68 = []
corr_hi68 = []


#plt.rc("text", usetex=True)
plt.rc("font", family="sans-serif")
plt.rcParams.update({"font.size": 8})

#f, ax1 = plt.subplots(1,1, figsize=(6.98167,6.98167*0.61803398875))
#f.set_size_inches(6.98167,6.98167*0.61803398875)

#f, ax1 = plt.subplots(1,1, figsize=(3.33,3.33*0.61803398875))
#f, ax1 = plt.subplots(1,1, figsize=(3.6,3.6*0.73803398875))
#f, ax1 = plt.subplots(1,1, figsize=(5.6,5.6*0.73803398875))
f, ax1 = plt.subplots(1,1, figsize=(3.6,3.6*0.73803398875))




plt.rc("font", family="sans-serif")
plt.rcParams.update({"font.size": 8})


symb = ["o", "s", "d", ">"] #["o", "s", "d"]
#LS = ["--", ":", ""]
#LS = [":", "--", "-.", "-" ]
#LS = ["-",":", "--", "-" ]
#cluster_type=["chiral"]
for i, ctype in enumerate(cluster_type):
	for j, folder in enumerate(N_values):

            #pfad = "./" + ctype + "_" + file1
            #pfad = "./" + ctype + "/" + subfolder + "/" + folders[i] + "/" + file1
            pfad = "./" + ctype + "/" + subfolder  + "/TRANS_CIS_CONTINUUM/" + file1
            print(pfad)
            #exit()

            #N, e_tot_avg, e_tot_std, e_vdwl_avg, e_vdwl_std, e_coul_avg, e_coul_std, e_hbi_avg, e_hbi_std  = np.loadtxt(pfad,skiprows=1, unpack=True)
            #corr, std, lo68, hi68, lo90, hi90, lo95, hi95 = np.loadtxt(pfad,skiprows=1, unpack=True)
            #tt, avg, cavg, avg_tot, cavg_tot = np.loadtxt(pfad,skiprows=3, unpack=True)
            tt, ccc, tcc, ttc, ttt = np.loadtxt(pfad,skiprows=1, unpack=True)
            ##print(corr)
            #print(corr.shape)
            N = N_values
            #print(N)
            #exit()

            #np.arange(1,Nmol)
            #Nmol.append(N)
            #E_avg.append(e_tot_avg)
            #E_std.append(e_tot_std)
            #corr_avg.append(corr)
            #corr_std.append(std)
            #corr_lo68.append(lo68)
            #corr_hi68.append(lo68)

            print(ttt.shape)
            
            t = np.arange(ttt.shape[0]) #/ 19.2  
            ws = 20 #100 # window size of rolling mean
            


            ttt_roll3 = rm3(ttt, ws)
            tttstd_roll3 = rm3e(ttt, ws)
            t_roll3 = rm3(t, ws)
            #print(t_roll3)
 
            #STD_roll3 = rm3(std,ws)
            #full_std = std_roll3 #STD_roll3+std_roll3
            # confidence intervals again?

            ttc_roll3 = rm3(ttc, ws)
            ttcstd_roll3 = rm3e(ttc, ws)

            tcc_roll3 = rm3(tcc, ws)
            tccstd_roll3 = rm3e(tcc, ws)

            ccc_roll3 = rm3(ccc, ws)
            cccstd_roll3 = rm3e(ccc, ws)

            #t_roll3 = rm3(t, ws)
            #print(t_roll3)


            #CSTD_roll3 = rm3(cstd,ws)

            #ax1.errorbar(np.arange(0,N), np.r_[np.array([1]),corr],  lw=1.0, mfc=palette[i], mec=palette[i], color=palette[i], fmt=symb[j], linestyle=LS[j], capsize=2., capthick=1., zorder=5, ms=3.)
            #ax1.errorbar(t, avg,  lw=1.0, mfc=palette[i], mec=palette[i], color=palette[i], fmt=symb[0], linestyle=LS[0], capsize=2., capthick=1., zorder=5, ms=3.)
            #ax1.plot(t, avg,  lw=1.0, color=palette[i], linestyle=LS[0], zorder=i, alpha=0.3)
            #ax1.plot(t_roll, avg_roll,  lw=1.0, color=palette[i], linestyle=LS[0], zorder=i+5, alpha=1)
            #ax1.plot(t_roll2, avg_roll2,  lw=1.0, color="black", linestyle=LS[0], zorder=i+5, alpha=0.5)
            ax1.plot(t_roll3, ttt_roll3,  lw=1.0, color=palette[0], linestyle="-", zorder=i+5, alpha=1., label="ttt")
            ax1.fill_between(t_roll3, ttt_roll3-tttstd_roll3, ttt_roll3+tttstd_roll3 ,alpha=0.15, facecolor=palette[0])

            ax1.plot(t_roll3, ttc_roll3,  lw=1.0, color=palette[1], linestyle="-", zorder=i+5, alpha=1., label="ttc")
            ax1.fill_between(t_roll3, ttc_roll3-ttcstd_roll3, ttc_roll3+ttcstd_roll3 ,alpha=0.15, facecolor=palette[1])

            ax1.plot(t_roll3, tcc_roll3,  lw=1.0, color=palette[2], linestyle="-", zorder=i+5, alpha=1., label="tcc")
            ax1.fill_between(t_roll3, tcc_roll3-tccstd_roll3, tcc_roll3+tccstd_roll3 ,alpha=0.15, facecolor=palette[2])

            ax1.plot(t_roll3, ccc_roll3,  lw=1.0, color=palette[3], linestyle="-", zorder=i+5, alpha=1., label="ccc")
            ax1.fill_between(t_roll3, ccc_roll3-cccstd_roll3, ccc_roll3+cccstd_roll3 ,alpha=0.15, facecolor=palette[3])



            p_tc = 0.0188
            p_ct = 0.0012
            ttt_pred = isomer_predict(3, p_tc, p_ct)
            ttc_pred = isomer_predict(2, p_tc, p_ct)
            tcc_pred = isomer_predict(1, p_tc, p_ct)
            ccc_pred = isomer_predict(0, p_tc, p_ct)
            print("ttt_pred, ttc_pred, tcc_pred, ccc_pred")
            print(ttt_pred, ttc_pred, tcc_pred, ccc_pred)


            #ax1.plot([-1, 501], [ttt_pred, ttt_pred], lw=1.0, color=palette[0], linestyle="--", zorder= 0) #label="ttt PSS")
            #ax1.plot([-1, 501], [ttc_pred, ttc_pred], lw=1.0, color=palette[1], linestyle=":", zorder= 0) #label="ttc PSS")
            #ax1.plot([-1, 501], [tcc_pred, tcc_pred], lw=1.0, color=palette[2], linestyle=(0, (3, 5, 1, 5)), zorder= 0) #label="tcc PSS")
            #ax1.plot([-1, 501], [ccc_pred, ccc_pred], lw=1.0, color=palette[3], linestyle=(0, (3, 5, 1, 5, 1, 5)), zorder= 0) #label="ccc PSS")

            ax1.plot([-1, 501], [ttt_pred, ttt_pred], lw=1.0, color=palette[0], linestyle=":", zorder= 0) #label="ttt PSS")
            ax1.plot([-1, 501], [ttc_pred, ttc_pred], lw=1.0, color=palette[1], linestyle=":", zorder= 0) #label="ttc PSS")
            ax1.plot([-1, 501], [tcc_pred, tcc_pred], lw=1.0, color=palette[2], linestyle=":", zorder= 0) #label="tcc PSS")
            ax1.plot([-1, 501], [ccc_pred, ccc_pred], lw=1.0, color=palette[3], linestyle=":", zorder= 0) #label="ccc PSS")

            
#Nmol = np.reshape(np.array(Nmol), (len(cluster_type), -1))

#print(Nmol)
#corr_avg = np.reshape(np.array(corr_avg), (len(cluster_type), -1)) 
#corr_std = np.reshape(np.array(corr_std), (len(cluster_type), -1))
#corr_lo68 = np.reshape(np.array(corr_lo68), (len(cluster_type), -1))
#corr_hi68 = np.reshape(np.array(corr_hi68), (len(cluster_type), -1))
#E_avg = np.reshape(np.array(E_avg),(len(cluster_type), -1))
#E_std = np.reshape(np.array(E_std),(len(cluster_type), -1))


#print(corr_avg)
#exit()

#for i, ctype in enumerate(cluster_type):
#    print(Nmol[i,:])
#    print(corr_avg[i,:])
#    exit()
#    ax1.errorbar(Nmol[i,:], corr_avg[i,:],  lw=1.0, mfc=palette[i], mec=palette[i], color=palette[i], fmt='o', linestyle=":", capsize=2., capthick=1., label=mylabel[i] + " (DREI)", zorder=5, ms=3.)
#ax1.plot([-10, 510], [1.,1.], c="xkcd:gray", lw=1., ls="--", zorder=-1)

ax1.set_ylim(bottom=-0.05, top=1.05)
ax1.set_xlim(left=0, right=500)


#ax1.legend(loc="best",frameon=False, ncol=2, handlelength=3)

ax1.set_xlabel(r'cycle', fontsize=12)

#plt.ylabel(r'$\langle E_\mathrm{bind} \rangle_t$ / (N - 1)')
#plt.ylabel(r'$\langle \vec{u}_i \cdot \vec{u}_j \rangle $')
plt.ylabel(r'Molar Fraction x', fontsize=12)




### numerical solution

pct =  0.0012
ptc =  0.0188

a = ptc #/100
b = pct #/100

#N = 1.08 # N azo

a1 = ptc*3
a2 = ptc*2
a3 = ptc

b1 = pct
b2 = pct*2
b3 = pct*3

from scipy.integrate import odeint


#def dydt(P, t):
#    coeff_matrix = ()
#    return 

def dx_dt(x, t):
    #return [-a*x[0] + b*x[1],   a*x[0] - (a+b)*x[1] + b * x[2], a*x[1] - (a+b)*x[2] + b * x[3], a *x[2] - b * x[3]]
    #return [-a*x[0] + b*x[1],  a*x[0] - (a+b)*x[1] + b * x[2], a*x[1] - (a+b)*x[2] + b * x[3], a *x[2] - b * x[3]]
    #return [-3*a*x[0] + b*x[1],  2*a*x[0] - (a+b)*x[1] + b * x[2], a*x[1] - (a+b)*x[2] + b * x[3], a *x[2] - b * x[3]]
    #return [-a1*x[0] + b1*x[1],  a1*x[0] - (a2+b1)*x[1] + b2 * x[2], a2*x[1] - (a3+b2)*x[2] + b3 * x[3], a3 *x[2] - b3 * x[3]]
    return [-3*a*x[0] + b*x[1],  3*a*x[0] - (2*a+b)*x[1] + 2*b * x[2], 2*a*x[1] - (a+2*b)*x[2] + 3*b*x[3], a*x[2] - 3*b*x[3]]


ts = np.linspace(0, 500, 5000)


x0 = [1.0, 0.0, 0.0, 0.0]

xs = odeint(dx_dt, x0, ts)
x_ttt = xs[:,0]
x_ttc = xs[:,1]
x_tcc = xs[:,2]
x_ccc = xs[:,3]


print(x_ttt[-1], x_ttc[-1], x_tcc[-1], x_ccc[-1])
print(x_ttt[-1]+ x_ttc[-1]+x_tcc[-1]+x_ccc[-1])

#plt.plot(ts, x_ttt, c = palette[0], ls="--", lw=1.)
#plt.plot(ts, x_ttc, c=palette[1], ls="--", lw=1.)
#plt.plot(ts, x_tcc, c =palette[2], ls="--", lw=1.)
#plt.plot(ts, x_ccc, c =palette[3], ls="--", lw=1.)




########################
#### exact solution ####
########################

def frac_ttt(t, a, b):
    return a**3 / (a+b)**3 * ((b**3)/(a**3) + 3*(b**2)/(a**2) * np.exp(-1.*(a+b)*t) + 3*b/a * np.exp(-2.*(a+b)*t)  + np.exp(-3.*(a+b)*t))

def frac_ttc(t, a, b):
    return a**3 / (a+b)**3 * ( 3*(b**2)/(a**2) + 3*(2*a*b - b**2)/(a**2) * np.exp(-1.*(a+b)*t) + 3*(a-2*b)/a* np.exp(-2.*(a+b)*t)  - 3 * np.exp(-3.*(a+b)*t))


def frac_tcc(t, a, b):
    return a**3 / (a+b)**3 * (3*b/a + 3*(a-2*b)/a * np.exp(-1.*(a+b)*t) + 3*(b-2*a)/a * np.exp(-2.*(a+b)*t)  + 3 * np.exp(-3.*(a+b)*t))


def frac_ccc(t, a, b):
    return a**3 / (a+b)**3 * (1  - 3 * np.exp(-1.*(a+b)*t) + 3 * np.exp(-2.*(a+b)*t) - 1 * np.exp(-3.*(a+b)*t))



ttt_vec = np.zeros(500)
ttc_vec = np.zeros(500)
tcc_vec = np.zeros(500)
ccc_vec = np.zeros(500)
print(ttt_vec)




for t in range(0,500):
    ttt_vec[t] = frac_ttt(t, a, b)
    ttc_vec[t] = frac_ttc(t, a, b)
    tcc_vec[t] = frac_tcc(t, a, b)
    ccc_vec[t] = frac_ccc(t, a, b)


T = np.arange(0,500)


plt.plot(T, ttt_vec,   c = palette[0], lw=1., ls="--")
plt.plot(T, ttc_vec,    c = palette[1], lw=1., ls="--")
plt.plot(T, tcc_vec,    c = palette[2], lw=1., ls="--")
plt.plot(T, ccc_vec,    c = palette[3], lw=1., ls="--")


plt.plot([-1,-1], [-1,-1], color="xkcd:black", linestyle="-", zorder=i+5, alpha=1., label="MD data", lw=1.)
ax1.plot([-1, -1], [-1, -1], lw=1.0, color="xkcd:black", linestyle="--", zorder= 0, label="theory")
ax1.plot([-1, -1], [-1, -1], lw=1.0, color="xkcd:black", linestyle=":", zorder= 0, label="PSS")

ax1.legend(loc="best",frameon=False, ncol=2, handlelength=3)

os.chdir("PLOTS")
plt.savefig("isomer_kinetics_predict_asym_COMPARE_THEORY2.png", bbox_inches= 'tight', dpi=300)
plt.savefig("isomer_kinetics_predict_asym_COMPARE_THEORY2.pdf", bbox_inches= 'tight', dpi=300)

plt.show()

#exit()







exit()

pfadttt1 = "../../xclusters/smaller/"
pfadttt2 = "../../xclusters/new_ttt/"
pfadttt3 =  "../../xclusters/2new_ttt/"


symb = ["o", "s", "d", ">"]
#LS = [":", "--", "-.", "-" ]
LS = [":", "--", "-" ]
N_values = [6, 10, 20]
folders = ["6", "10",  "20"]

subfolder = "rerun_chain"

cluster_type = [pfadttt1, pfadttt2, pfadttt3]
#corr_avg = []
corrs = []
#for i, ctype in enumerate(cluster_type):
for j, folder in enumerate(folders):
    corrs = []
    for i, ctype in enumerate(cluster_type):

            pfad = ctype + folder + "/" + subfolder + "/" + file1
            print(pfad)
            corr, std, lo68, hi68, lo90, hi90, lo95, hi95 = np.loadtxt(pfad,skiprows=1, unpack=True)
            N = N_values[j]

            print(corr)

            corrs.append(corr)

    print("CORRS")
    print(corrs)
    corr_avg = np.mean(corrs, axis=0)
    print("CORR_AVG")
    print(corr_avg)
    YY=np.r_[np.array([1]),corr_avg]
    print(YY)
    therange=np.arange(0,N)
    print(N)
    print(therange.shape)
    print(YY.shape)
    #exit()
    

    #ax1.errorbar(np.arange(0,N), np.r_[np.array([1]),corr_avg],  lw=1.0, mfc=palette[0], mec=palette[0], color=palette[0], fmt=symb[j], linestyle=LS[j], capsize=2., capthick=1., zorder=5, ms=3.)
    ax1.errorbar(np.arange(0,N), YY,  lw=1.0, mfc=palette[0], mec=palette[0], color=palette[0], fmt=symb[j], linestyle=LS[j], capsize=2., capthick=1., zorder=5, ms=3.)





##### CCC ###########


pfadccc1 = "../../xclusters/new_ccc/"
pfadccc2 = "../../xclusters/2new_ccc/"


symb = ["o", "s", "d", ">"]
#LS = [":", "--", "-.", "-" ]
LS = [":", "--", "-" ]
N_values = [6, 10, 20]
folders = ["6", "10",  "20"]

subfolder = "rerun_chain"

cluster_type = [pfadccc1, pfadccc2]
#corr_avg = []
corrs = []
#for i, ctype in enumerate(cluster_type):
for j, folder in enumerate(folders):
    corrs = []
    for i, ctype in enumerate(cluster_type):

            pfad = ctype + folder + "/" + subfolder + "/" + file2
            print(pfad)
            corr, std, lo68, hi68, lo90, hi90, lo95, hi95 = np.loadtxt(pfad,skiprows=1, unpack=True)
            N = N_values[j]

            print(corr)

            corrs.append(corr)


    corr_avg = np.mean(corrs, axis=0)

    ax1.errorbar(np.arange(0,N), np.r_[np.array([1]),corr_avg],  lw=1.0, mfc=palette[2], mec=palette[2], color=palette[2], fmt=symb[j], linestyle=LS[j], capsize=2., capthick=1., zorder=5, ms=3.)




### legend ####

folders = ["6", "10", "20"]
ax1.errorbar([-10,-10], [-10,-10], yerr=[1,1],  lw=1.0, mfc="xkcd:black", mec="xkcd:black", color="xkcd:black" , fmt=symb[0], linestyle=LS[0], capsize=2., capthick=1., zorder=5, ms=3., label = "N = "+ folders[0])
ax1.errorbar([-10,-10], [-10,-10], yerr=[1,1], lw=1.0, mfc="xkcd:black", mec="xkcd:black", color="xkcd:black" , fmt=symb[1], linestyle=LS[1], capsize=2., capthick=1., zorder=5, ms=3., label = "N = "+ folders[1])
ax1.errorbar([-10,-10], [-10,-10], yerr=[1,1],lw=1.0, mfc="xkcd:black", mec="xkcd:black", color="xkcd:black" , fmt=symb[2], linestyle=LS[2], capsize=2., capthick=1., zorder=5, ms=3., label = "N = "+ folders[2])
#ax1.errorbar([-10,-10], [-10,-10], yerr=[1,1], lw=1.0, mfc="xkcd:black", mec="xkcd:black", color="xkcd:black" , fmt=symb[3], linestyle=LS[3], capsize=2., capthick=1., zorder=5, ms=3., label = "N = "+ folders[3])

ax1.errorbar([-10,-10], [-10,-10], yerr=[1,1], lw=1.0, mfc=palette[0], mec=palette[0], color=palette[0], fmt=symb[0], linestyle=LS[0], capsize=2., capthick=1., zorder=5, ms=3., label = "ttt")
ax1.errorbar([-10,-10], [-10,-10], yerr=[1,1], lw=1.0, mfc=palette[2], mec=palette[2], color=palette[2], fmt=symb[0], linestyle=LS[0], capsize=2., capthick=1., zorder=5, ms=3., label = "ccc")
#ax1.errorbar([-10,-10], [-10,-10], yerr=[1,1], lw=1.0, mfc=palette[1], mec=palette[1], color=palette[1], fmt=symb[0], linestyle=LS[0], capsize=2., capthick=1., zorder=5, ms=3., label = "II")
#ax1.errorbar([-10,-10], [-10,-10], yerr=[1,1], lw=1.0, mfc=palette[2], mec=palette[2], color=palette[2], fmt=symb[0], linestyle=LS[0], capsize=2., capthick=1., zorder=5, ms=3., label = "III")

#ax1.legend(loc="best",frameon=False, ncol=2)

ax1.set_xlabel(r'$|i-j|$', fontsize=12)

#plt.ylabel(r'$\langle E_\mathrm{bind} \rangle_t$ / (N - 1)')
#plt.ylabel(r'$\langle \vec{u}_i \cdot \vec{u}_j \rangle $')
plt.ylabel(r'$\langle \vec{u}_i \cdot \vec{u}_{i + |i-j|} \rangle $', fontsize=12)
#ax1.set_ylabel(r'$\Delta r_\mathrm{pairs}$ [$\mathrm{\AA}$]')
#ax1.set_xlim(left=0.0,right=22.)
#ax1.set_ylim(top=1.2,bottom=0.)

#ax1.set_ylim(top=-30.0,bottom=-65.)

#ax1.set_xlim(left=0.0, right= 42)
#plt.xticks([0, 6, 12, 18, 24, 30, 36, 42])

ax1.set_xlim(left=0.0, right= 36)
ax1.set_ylim(top=1.05, bottom= 0)
#plt.xticks([1, 5, 10, 15, 20, 25, 30, 35])


#plt.title("60° Twist")

os.chdir("../../PLOT_CLUSTERS/")




plt.savefig("Correlation_orient_avg_ttt_ccc_alt.png", bbox_inches= 'tight', dpi=300)
plt.savefig("Correlation_orient_avg_ttt_ccc_alt.pdf", bbox_inches= 'tight', dpi=300)


ax1.legend(loc="best",frameon=False, ncol=2)

ax1.set_xlabel(r'$|i-j|$', fontsize=12)

#plt.ylabel(r'$\langle E_\mathrm{bind} \rangle_t$ / (N - 1)')
#plt.ylabel(r'$\langle \vec{u}_i \cdot \vec{u}_j \rangle $')
plt.ylabel(r'$\langle \vec{u}_i \cdot \vec{u}_{i + |i-j|} \rangle $', fontsize=12)
#ax1.set_ylabel(r'$\Delta r_\mathrm{pairs}$ [$\mathrm{\AA}$]')
#ax1.set_xlim(left=0.0,right=22.)
#ax1.set_ylim(top=1.2,bottom=0.)

#ax1.set_ylim(top=-30.0,bottom=-65.)

#ax1.set_xlim(left=0.0, right= 42)
#plt.xticks([0, 6, 12, 18, 24, 30, 36, 42])

ax1.set_xlim(left=0, right= 6.5)
plt.xticks([0, 1, 2,3,4,5,6])
ax1.set_ylim(top=1.05, bottom= 0)

plt.savefig("Correlation_orient_avg_PCFF_36_PART_smaller_select_NEW_alt2.png", bbox_inches= 'tight', dpi=300)
plt.savefig("Correlation_orient_avg_PCFF_36_PART_smaller_select_NEW_alt2.pdf", bbox_inches= 'tight', dpi=300)

plt.show()

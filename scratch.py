

Nlat = len(ds.latitude)
Nlon = len(ds.longitude)

Ntime = len(ds.time)
Npoints = Nlat*Nlon
u = np.reshape(ds.uo.values, (Ntime, Npoints)).T
v = np.reshape(ds.vo.values, (Ntime, Npoints)).T

reorder = np.vstack([np.arange(Npoints),
                     np.arange(Npoints, 2*Npoints)]).reshape((-1,),
                                                             order='F')


X = np.vstack([u,v])
X = X[reorder,:]


Xkp1 = X[:,2:]
Xk = X[:,1:-1]
Xkm1 = X[:,:-2]

# secant predictor
Uk = 2*Xk - Xkm1

u_pr,v = get_uv(Uk)
u_tr,v = get_uv(Xkp1)
diff = np.abs(u_tr-u_pr)


plt.close('all');
plt.subplot(2,2,1)
plt.imshow(u_pr[100,:,:]);
plt.gca().invert_yaxis();
plt.subplot(2,2,2)
plt.imshow(u_tr[100,:,:]);
plt.gca().invert_yaxis();
plt.subplot(2,2,3)
plt.imshow(diff[100,:,:]);
plt.gca().invert_yaxis();
plt.pause(1);


def get_uv(X):
    Nt = X.shape[1]
    u = np.reshape((X[0::2, :]).T, (Nt, Nlat, Nlon))
    v = np.reshape((X[1::2, :]).T, (Nt, Nlat, Nlon))    
    return u,v

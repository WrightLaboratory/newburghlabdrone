from __future__ import print_function, division 
import numpy as np
try:
    import matplotlib.pyplot as plt
    import  matplotlib.colors as colors
except:
    pass
import sys
from numpy.fft import rfft, irfft, ifft
from scipy.optimize import leastsq
import numpy.lib.recfunctions as rf
import os

class BMXFile(object):
    freqOffset = 1100
    def __init__(self,fname, nsamples=None, force_version=None, loadD2=False, loadRFI=False, verbose=0):
        ## old header!!
        #head_desc=[('nChan','i4'),
        #           ('fftsize','i4'),('fft_avg','i4'),('sample_rate','f4'),
        #           ('numin','f4'),('numax','f4'),('pssize','i4')]

        self.verbose=verbose
        prehead_desc=[('magic','S8'),('version','i4')]
        f=open(fname);
        H=np.fromfile(f,prehead_desc,count=1)
        if H['magic'][:7]!=b'>>BMX<<':
            print("Bad magic.",H['magic'])
            sys.exit(1)
        if force_version is not None:
            self.version=force_version
        else:
            self.version=H['version'][0]
        if verbose>0:
            print ("Loading: ",fname)
            print ("Format Version:",self.version)
        if verbose>1:
            print ("Header:",H)
        if self.version<=5:
            maxcuts=10
            head_desc=[('nChan','i4'),('sample_rate','f4'),('fft_size','u4'),
                   ('ncuts','i4'),
                   ('numin','10f4'),('numax','10f4'),('fft_avg','10u4'),
                   ('pssize','10i4')]
        elif self.version<=6:
            maxcuts=10
            head_desc=[('cardMask','i4'),('nChan','i4'),('sample_rate','f4'),
                       ('fft_size','u4'),('ncuts','i4'),
                   ('numin','10f4'),('numax','10f4'),('fft_avg','10u4'),
                   ('pssize','10i4')]
        elif self.version<=7:
            maxcuts=10
            head_desc=[('cardMask','i4'),('nChan','i4'),('sample_rate','f4'),
                       ('fft_size','u4'),('average_recs','u4'), ('ncuts','i4'),
                   ('numin','10f4'),('numax','10f4'),('fft_avg','10u4'),
                   ('pssize','10i4'),('bufdelay','2i4'),('delay','2i4')]
        elif self.version<=8:
            maxcuts=10
            head_desc=[('daqNum','i4'), ('wires','S8'),
                       ('cardMask','i4'),('nChan','i4'),('sample_rate','f4'),
                       ('fft_size','u4'),('average_recs','u4'), ('ncuts','i4'),
                       ('numin','10f4'),('numax','10f4'),('fft_avg','10u4'),
                       ('pssize','10i4'),('bufdelay','2i4'),('delay','2i4')]
            
        else:
            print ("Unknown version",H['version'])
            sys.exit(1)
        H=np.fromfile(f,head_desc,count=1)
        if self.version>=6: self.cardMask=H['cardMask']
        self.nChan=H['nChan'][0]
        self.ncuts=H['ncuts'][0]
        if self.version>=6: 
            if verbose>0: print ("CardMask: %i, Channels: %i,  Cuts: %i"%(self.cardMask, self.nChan, self.ncuts))
        else: 
            print("Channels: %i,  Cuts: %i"%(self.nChan, self.ncuts))
        self.fft_size=H['fft_size'][0]
        self.average_recs=H['average_recs'][0] if self.version>=7 else 1
        self.sample_rate=H['sample_rate']/1e6
        self.deltaT = 1./self.sample_rate*self.fft_size/1e6 * self.average_recs
        self.nP=H['pssize'][0]
        self.numin=(H['numin'][0]/1e6)[:self.ncuts]
        self.numax=(H['numax'][0]/1e6)[:self.ncuts]
        if verbose>0: print("We have ",self.ncuts,"cuts:")
        self.freq=[]
        for i in range(self.ncuts):
            if verbose>0: print("    Cut ",i," ",self.numin[i],'-',self.numax[i],'MHz #P=',self.nP[i])
            self.freq.append(self.freqOffset+self.numin[i]+(np.arange(self.nP[i])+0.5)*(self.numax[i]-self.numin[i])/self.nP[i])
        rec_desc=[]
        self.haveMJD=False
        self.haveNulled=False
        self.haveToneFreq=False
        self.haveDiode=False
        self.FilenameUTC=(self.version>=5)

        if self.version>=8:
            self.wires={} ## let's make this dictonary to avoid counting convention crap
            j=1
            for i in range(0,len(H['wires'][0]),2):
                self.wires[j]=H['wires'][0][i:i+2]
                j+=1
        if self.version>=7:
            self.delay=H['delay'][0]
            self.bufdelay=H['bufdelay'][0]
        if self.version>=3:
            self.haveMJD=True
            rec_desc+=[('mjd','f8')]
        if self.version>=2:
            rec_desc+=[('num_nulled','i4',H['nChan'])]
            self.haveNulled=True

        self.nCards=(1+int(H['cardMask']==3)) if (self.version>=6) else 1;
        self.nChanTot=self.nCards*self.nChan
        if (self.nCards==1): 
            if self.nChan==1:
                for i in range(self.ncuts):
                    rec_desc+=[('chan1_'+str(i),'f4',self.nP[i])]
            else:
                for i in range(self.ncuts):
                    rec_desc+=[('chan1_'+str(i),'f4',self.nP[i]),
                               ('chan2_'+str(i),'f4',self.nP[i]), 
                               ('chan12R_'+str(i),'f4',self.nP[i]),
                               ('chan12I_'+str(i),'f4',self.nP[i])]
        else:
            if self.nChan==1:
                print ("This is not possible")
                throw
            else:
                for i in range(self.ncuts):
                    for ch in range(1,5):
                        rec_desc+=[('chan%i_%i'%(ch,i),'f4',self.nP[i])]
                    for ch1 in range(1,5):
                        for ch2 in range(ch1+1,5):
                               rec_desc+=[('chan%ix%iR_%i'%(ch1,ch2,i),'f4',self.nP[i]),
                                          ('chan%ix%iI_%i'%(ch1,ch2,i),'f4',self.nP[i])]

        if self.version>=1.5:
            rec_desc+=[('nu_tone','f4')]
            self.haveToneFreq=True
        if self.version>=4:
            rec_desc+=[('lj_voltage','f4'),('lj_diode','i4')]
            self.haveDiode=True

        rec_dt=np.dtype(rec_desc,align=False)
        self.rec_dt=rec_dt
        if nsamples is None:
            self.data=np.fromfile(f,rec_dt)
        else:
            self.data=np.fromfile(f,rec_dt,count=nsamples)
        self.fhandle=f
        self.nSamples = len(self.data)
        if verbose>0: print ("Loading done, %i samples"%(len(self.data)))
        self.names=self.data.dtype.names
        self.fname = fname
        if loadRFI:
            if verbose>0: print('Loading RFI...')
            rfierr = self.loadRFI()
            if rfierr: loadRFI = False
        if loadD2:
            D2File=BMXFile(fname.replace("D1","D2"),
                           nsamples=nsamples, force_version=force_version, loadD2=False, 
                           loadRFI=loadRFI, verbose=verbose)
            self.joinD2(D2File, loadRFI=loadRFI)
        self.names=self.data.dtype.names
        self.nSamples = self.data['chan1_0'].shape[0]

    def joinD2(self,D2,loadRFI=False):
        ## set num channels to 8
        self.nChanTot+=D2.nChanTot
        L=min(len(self.data),len(D2.data))
        ## First find best offset, starting with zero
        offset=0
        def getSlices(offset, d1data=self.data, d2data=D2.data):
            if offset>0:
                slice1=d1data[offset:]
                slice2=d2data[:]
                L=min(len(slice1),len(slice2))
                slice1=slice1[:L]
                slice2=slice2[:L]
            else:
                slice1=d1data[:]
                slice2=d2data[-offset:]
                L=min(len(slice1),len(slice2))
                slice1=slice1[:L]
                slice2=slice2[:L]
            return slice1, slice2
        
        def mjdoffset(offset):
            s1,s2=getSlices(offset)
            return s2['mjd'].mean()-s1['mjd'].mean()

        while True:
            p0,pp,pm=mjdoffset(offset)**2,mjdoffset(offset+1)**2,mjdoffset(offset-1)**2
            if p0<pp and p0<pm:
                break
            elif pm<pp:
                offset-=1
            else:
                offset+=1
            if (abs(offset)>L-2):
                print ("Offset not converging!")
                raise RuntimeError("Offset")
        self.dTD2=mjdoffset(offset)*3600*24.
        if (self.verbose>0):
            print ("Best offset:",offset, 'dTD2:',self.dTD2,"s")
        data1,data2=getSlices(offset)
        data1=rf.drop_fields(data1,['lj_voltage','lj_diode','nu_tone'])
        data2=rf.drop_fields(data2,['mjd','nu_tone'])
        # rename fields in 12
        d1,d2={},{}
        for n in data1.dtype.names:
            if 'chan' in n:
                #d1[n]=n.replace('chan','chanA')
                
                d2[n]=n.replace('1','5').replace('2','6').replace('3','7').replace('4','8')
        d1['num_nulled']='num_nulled14'
        d2['num_nulled']='num_nulled58'

        data1=rf.rename_fields(data1,d1)
        data2=rf.rename_fields(data2,d2)
        self.data=rf.merge_arrays((data1,data2),flatten=True)
        if loadRFI:
            rfi1, rfi2 = getSlices(offset, d1data=self.rfi, d2data=D2.rfi)
            rfimask1, rfimask2 = getSlices(offset, d1data=self.rfimask, d2data=D2.rfimask)
            rfinumbad1, rfinumbad2 = getSlices(offset, d1data=self.rfinumbad, d2data=D2.rfinumbad)
            rfid2={}
            for n in data1.dtype.names:
                if 'chan' in n:
                    rfid2[n]=n.replace('1','5').replace('2','6').replace('3','7').replace('4','8')
            rfi2=rf.rename_fields(rfi2,rfid2)
            rfimask2=rf.rename_fields(rfimask2,rfid2)
            rfinumbad2=rf.rename_fields(rfinumbad2,rfid2)
            self.rfi=rf.merge_arrays((rfi1,rfi2),flatten=True)
            self.rfimask=rf.merge_arrays((rfimask1,rfimask2),flatten=True)
            self.rfinumbad=rf.merge_arrays((rfinumbad1,rfinumbad2),flatten=True)
        if (self.verbose>0):
            print ("Merge successful, total records:",len(self.data))

    def update(self,replace=False):
        ndata=np.fromfile(self.fhandle,self.rec_dt)
        nd=len(ndata)
        if replace:
            self.data=ndata
        else:
            self.data=np.hstack((self.data,ndata))
        return nd

    def getNames(self, cut):
        cutstr="_"+str(cut)
        i=len(cutstr)
        chnames=sorted([n for n in self.names if ('chan' in n) and (n[-i:]==cutstr)])
        return chnames

    def getAutoName(self,chan,cut):
        return 'chan%i_%i'%(chan,cut)
    
    def getRadar (self,fmin=1242., fmax=1247.):
        imin,imax,_,_=self.f2iminmax(fmin,fmax)
        da=(self.data[:]['chan1_0'])[:,imin:imax]+(self.data[:]['chan2_0'])[:,imin:imax]
        da=da.mean(axis=1)
        self.radarOn=(da>da.mean()) ## this seems to work pretty well!

    def filterRadar(self):
        ## fill in radar sections with neighbors
        for i,r in enumerate(self.radarOn):
            if r:
                ## find the closest non-radar point
                l=i-1
                h=i+1
                while True:
                    if not(self.radarOn[l]):
                        j=l
                        break
                    if not(self.radarOn[h]):
                        j=h
                        break
                    if (l>0):
                        l-=1
                    if (h<self.nSamples):
                        h+=1
                self.data[i]=self.data[j]

    def normalizeOnRegion(self,fmin,fmax,cut=0):
        imin,imax,_,_=self.f2iminmax(fmin,fmax)

        for ch in [1,2]:
            name=self.getAutoName(ch,cut)
            nfactor=(self.data[:][name])[:,imin:imax].mean(axis=1)
            for i in range(self.nSamples):
                self.data[i][name]/=nfactor[i]

    def f2iminmax(self,fmin,fmax,cut=0,binSize=1):
        if fmin is None:
            imin=0
            fmin=self.freq[cut][0]
        else:
            imin=(self.freq[cut]>fmin).argmax()-1
            if (imin<0):
                imin=0
            fmin=self.freq[cut][imin]
        if fmax is None:
            imax=len(self.freq[cut])
            fmax=self.freq[cut][-1]
        else:
            imax=(self.freq[cut]<fmax).argmin()+1
            if (imax-imin)%binSize!=0:
                imax-=(imax-imin)%binSize+binSize
            if imax>=len(self.freq[cut]):
                imax=len(self.freq[cut])-1
            fmax=self.freq[cut][imax]
        return imin,imax,fmin,fmax
    
    def plotAvgSpec(self, cut=0):
        for i, n in enumerate(self.getNames(cut)):
            plt.subplot(2,2,i+1)
            y=self.data[n].mean(axis=0)
            plt.plot(self.freq[cut],y)
            plt.xlabel('freq [MHz] ' + n)
   
    #waterfall plot of frequencies over time. Can either use log scale, or subtract and divide off the mean 
    def plotWaterfall(self, fmin=None, fmax=None, nsamples=None, cut=0, binSize = 4, subtractMean = False, minmax=None):
        if nsamples is  None:
            nsamples = self.nSamples  #plot all samples in file
        imin,imax,fmin,fmax=self.f2iminmax(fmin,fmax,cut,binSize)
        
        for n in range(2):
           plt.subplot(2,1, n+1)
           arr = []
           for i in range(nsamples):
               arr.append(self.data[i]['chan' + str(n+1)+'_' + str(cut)][imin:imax])
               arr[i] = np.reshape(arr[i],(-1, binSize )) #bin frequencies
               arr[i] = np.mean(arr[i], axis = 1)  #average the bins
           arr=np.array(arr)
           if(subtractMean):
               means = np.mean(arr, axis=0) #mean for each freq bin
               for j in range(nsamples):
                   arr[j,:] -= means
                   arr[j,:] /=means
               if minmax is not None:
                   vmin = -minmax
                   vmax = minmax
               else:
                   vmin = None
                   vmax = None
               plt.imshow(arr, interpolation="nearest", vmin=vmin,vmax=vmax, aspect = "auto", extent=[fmin, fmax, nsamples*self.deltaT, 0])
           else:
               plt.imshow(arr, norm=colors.LogNorm(), interpolation="nearest" , aspect = "auto", extent=[fmin, fmax, nsamples*self.deltaT, 0]) 
           plt.colorbar()
           plt.xlabel('freq [MHz] Channel ' + str(n+1))
           plt.ylabel('time [s]')
        plt.show()


    #return image array to use as input for numpy.imshow
    #inputs:
    #       binSize [x, y]: how many samples to average per bin on both the x and y  axis
    #       cut: which cut to use
    def getImageArray(self, binSize = [1, 1], nsamples=None, cut=None ):
        if nsamples is  None:
            nsamples = self.nSamples
        reducedArr = []  #for reduced array after binning
        if cut is None:
            cut='chan1_0'
        arr = []
        for i in range(nsamples):
            arr.append(self.data[i][cut])
            #bin along x axis (frequency bins)
            if binSize[0] > 1:
                arr[i] = np.reshape(arr[i],(-1, binSize[0])) 
                arr[i] = np.mean(arr[i], axis = 1)  
                
        arr = np.array(arr)
 
        #bin along y axis (time bins)
        if binSize[1] > 1:
            reducedArr.append([])
            for i in range(int(len(arr)/binSize[1])):
                reducedArr[n].append(arr[binSize[1]*i:binSize[1]*(i+1)].mean(axis=0))
        else:
            reducedArr = arr
 
        return (reducedArr)


    def nullBin(self, bin):
        for cut in [0]:
            for i, n in enumerate(self.getNames(cut)):
                self.data[n][:,bin]=0.0
                
    def getToneAmplFreq(self,chan, pm=20,freq="index"):
        mxf=[]
        mx=[]
        for line in self.data[chan]: 
            i=abs(line).argmax()
            mxf.append(i)
            mx.append(line[max(0,i-pm):i+pm].sum())
        mx=np.array(mx)
        if freq=="index":
            mxf=np.array(mxf)
        elif freq=="freq" or "dfreq":
            mxf=np.array([self.freq[chan][i] for i in mxf])
            if freq=="dfreq":
                mxf-=self.freq[chan].mean()
        return mxf,mx

    def parseRFI(self, fname):
        magic_desc = [('magic','S8')]
        prehead_desc = [('totbad','i2')]
        head_desc = [('ind','i2'),('num','i2'),('val','f4')]
        rfi = np.zeros((self.nSamples, 16*2048), dtype=np.float32)
        rfimask = np.zeros((self.nSamples, 16*2048), dtype=np.float32)
        numbad = np.zeros((self.nSamples, 16*2048), dtype=np.float32)
        
        f = open(fname)
        H = np.fromfile(f, magic_desc, count=1)[0]
        # RFI version 1
        if H['magic'][:7]==b'>>RFI<<':
            preprehead_desc = [('nSigma','f4')]
            nSigma = np.fromfile(f, preprehead_desc, count=1)[0][0]
            for i in range(self.nSamples):
                lastind = 0
                totbad = np.fromfile(f, prehead_desc, count=1)[0][0]
                while totbad>0:
                    H = np.fromfile(f, head_desc, count=1)[0]
                    rfi[i,H['ind']] = H['val']
                    rfimask[i,H['ind']] = 1
                    numbad[i,H['ind']] = H['num']
                    lastind = H['ind']
                    totbad += -H['num']
        # RFI version 2
        elif H['magic'][:7]==b'>>RFI2<':
            preprehead_desc = [('version','i4'),('nSigma','f4')]
            H = np.fromfile(f, preprehead_desc, count=1)
            for i in range(self.nSamples):
                totbad = np.fromfile(f, prehead_desc, count=1)[0][0]
                H = np.fromfile(f, head_desc, count=totbad)
                rfi[i,H['ind']] = H['val']
                rfimask[i,H['ind']] = 1
                numbad[i,H['ind']] = H['num']
        else:
            print("Bad magic.",H['magic'])
            sys.exit(1)
        f.close()
        return rfi, rfimask, numbad

    def loadRFI(self, fname_rfi=None):
        # Load data from file
        if fname_rfi is None:
            fname = self.fname
            fname_rfi = os.path.join(os.path.dirname(fname),'rfi',os.path.basename(fname).replace('data','rfi'))
        if not os.path.isfile(fname_rfi): # if RFI file does not exist, exit with error
            print('RFI file does not exist.')
            return 1
        rfi, rfimask, numbad = self.parseRFI(fname_rfi)
        # Define channels
        dtype = []
        for name in self.names:
            if 'chan' in name:  
                 dtype.append((name,'2048f4'))
        # Restructure data into channels
        rfi = rfi.view(dtype=dtype)[:,0]
        rfimask = rfimask.view(dtype=dtype)[:,0]
        numbad = numbad.view(dtype=dtype)[:,0]
        # Save data
        self.rfi = rfi
        self.rfimask = rfimask
        self.rfinumbad = numbad
        return 0

            


class BMXRingbuffer (object):
    def __init__ (self,fname,force_version=None):
        prehead_desc=[('magic','S8'),('version','i4')]
        f=open(fname);
        H=np.fromfile(f,prehead_desc,count=1)
        if H['magic'][:7]!=b'>>RBF<<':
            print("Bad magic.",H['magic'])
            sys.exit(1)
        if force_version is not None:
            self.version=force_version
        else:
            self.version=H['version'][0]
        print ("Loading version:",self.version)
        head_desc=[('ncards','i4'),('size','i8')]
        H=np.fromfile(f,head_desc,count=1)
        self.ncards=H['ncards'][0]
        self.size=H['size'][0]//2 ## two channels
        print ("Ncards: %i Size: %i"%(self.ncards,self.size))
        data=np.fromfile(f,np.dtype([('ch1','i1'),('ch2','i1')],align=False))
        assert(len(data)==self.size*self.ncards)
        self.datad0c1=data['ch1'][:self.size]
        self.datad0c2=data['ch2'][:self.size]
        if (self.ncards>1):
            self.datad1c1=data['ch1'][self.size:]
            self.datad1c2=data['ch2'][self.size:]
            self.data=[self.datad0c1,self.datad0c2,self.datad1c1,self.datad1c2]
        else:
            self.data=[self.datad0c1,self.datad0c2]
        del data
        
    def getXis(self, start=0, end=2**28, maxlen=None,pad=False):
        #print ("FFTing...")
        size=end-start
        wspace=np.zeros(size*(1+int(pad)),np.float32)
        ffts=[]
        for i,d in enumerate(self.data):
            wspace[:size]=d[start:end].astype(np.float32)
            ffts.append(rfft(wspace))
        #print ("Calculating Xis")
        xi=[]
        for i,f1 in enumerate(ffts):
            for j,f2 in enumerate(ffts):
                if (j>i):
                    pass
                else:
                    corr=irfft(f1*np.conj(f2))
                    xi.append((i,j,corr))
        return xi
    
    def getOffsets(self, start=0, end=2**28):
        self.data[0]=self.data[1][30:]
        self.data[2]=self.data[1][15:]
        self.data[3]=self.data[1][5:]
        xi=self.getXis(start=start, end=end,pad=False)
        s=len(xi[0][2])
        s2=s//2
        maxc=[]
        for i,j,corr in xi:
            o=corr.argmax()
            if (o>s2):
                o-=s
            maxc.append([i,j,o])
        print (maxc)
        def mfun(ofs):
            return np.array([ofs[j]-ofs[i]-o for i,j,o in maxc])
        res,ie=leastsq(mfun, np.zeros(2*self.ncards))
        chi2=mfun(res)
        res-=res[1]
        print (res,chi2.sum())




            
                
        
        
        

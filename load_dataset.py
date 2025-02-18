

import numpy as np
from netCDF4 import Dataset

def read_nc(netcdf_file):
    ncid = Dataset(netcdf_file, 'r')

    nc_vars = [var for var in ncid.variables]
    for var in nc_vars:
        if hasattr(ncid.variables[str(var)], 'add_offset'):
            exec('global ' + str(var) + "; offset=ncid.variables['" + str(var) + "'].add_offset; " + str(
                var) + "=ncid.variables['" + str(var) + "'][:]-offset")
        else:
            exec('global ' + str(var) + '; ' + str(var) + "=ncid.variables['" + str(var) + "'][:]")
    ncid.close()
    return



#################read input/output data###########################
# model_dir='C:/Users/Michela/Desktop/LEZIONI PARTENOPE/progs/'
model_dir = './'
file_train = model_dir+'BIOARGO_MATCHUP_2018_2021.nc'


def load_dataset(file_train):
        # read_nc(file_train)
        
        ncid = Dataset(file_train, 'r')

        LAT = np.array(ncid['LAT'])
        LON = np.array(ncid['LON'])
        DOY = np.array(ncid['DOY'])
        SST = np.array(ncid['SST'])
        SSS = np.array(ncid['SSS'])
        SSD = np.array(ncid['SSD'])
        CHLSURF = np.array(ncid['CHLSURF'])
        ADT = np.array(ncid['ADT'])
        UGOS = np.array(ncid['UGOS'])
        VGOS = np.array(ncid['VGOS'])
        T = np.array(ncid['T'])
        S = np.array(ncid['S'])
        CHL = np.array(ncid['CHL'])
        DS = np.array(ncid['DS'])



        ########################################################################################################################

        random_array = np.random.choice(LAT.shape[0],LAT.shape[0], replace=False)
        # random_array=np.load(model_dir+'random_indices.npy')
        LATtot=LAT[random_array,:]
        LONtot=LON[random_array,:]
        JDtot=DOY[random_array,:]
        SSTtot=SST[random_array,:]
        SSStot=SSS[random_array,:]
        SSDtot=SSD[random_array,:]
        OWPtot=CHLSURF[random_array,:]
        ADTtot=ADT[random_array,:]
        UGOStot=UGOS[random_array,:]
        VGOStot=VGOS[random_array,:]

        Ttot=T[random_array,:]
        Stot=S[random_array,:]
        CHLtot=CHL[random_array,:]
        DStot=DS[random_array,:]

        ############divide the dataset in 80% for training and 20% for test#######################################################
        xx=int(LATtot.shape[0]*80/100)

        LATtraining=LATtot[0:xx,:]
        LONtraining=LONtot[0:xx,:]
        JDtraining=JDtot[0:xx,:]
        SSTtraining=SSTtot[0:xx,:]
        SSStraining=SSStot[0:xx,:]
        SSDtraining=SSDtot[0:xx,:]
        OWPtraining=OWPtot[0:xx,:]
        ADTtraining=ADTtot[0:xx,:]
        UGOStraining=UGOStot[0:xx,:]
        VGOStraining=VGOStot[0:xx,:]

        Ttraining=Ttot[0:xx,:]
        Straining=Stot[0:xx,:]
        CHLtraining=CHLtot[0:xx,:]
        DStraining=DStot[0:xx,:]

        #test dataset
        LATtest=LATtot[xx:,:]
        LONtest=LONtot[xx:,:]
        JDtest=JDtot[xx:,:]
        SSTtest=SSTtot[xx:,:]
        SSStest=SSStot[xx:,:]
        SSDtest=SSDtot[xx:,:]
        OWPtest=OWPtot[xx:,:]
        ADTtest=ADTtot[xx:,:]
        UGOStest=UGOStot[xx:,:]
        VGOStest=VGOStot[xx:,:]

        Ttest=Ttot[xx:,:]
        Stest=Stot[xx:,:]
        CHLtest=CHLtot[xx:,:]
        DStest=DStot[xx:,:]

        ############IDENTIFY THE MIN/max OF THE TRAINING DATASET##################################################

        LATmin=np.min(LATtraining)
        LONmin=np.min(LONtraining)
        SSTmin=np.min(SSTtraining)
        SSSmin=np.min(SSStraining)
        SSDmin=np.min(SSDtraining)

        OWPmin=np.min(np.log10(OWPtraining))
        ADTmin=np.min(ADTtraining)
        UGOSmin=np.min(UGOStraining)
        VGOSmin=np.min(VGOStraining)

        Tmin=np.min(Ttraining)
        Smin=np.min(Straining)
        CHLmin=np.min(np.log10(CHLtraining))
        DSmin=np.min(DStraining)


        LATmax=np.max(LATtraining)
        LONmax=np.max(LONtraining)
        SSTmax=np.max(SSTtraining)
        SSSmax=np.max(SSStraining)
        SSDmax=np.max(SSDtraining)

        OWPmax=np.max(np.log10(OWPtraining))
        ADTmax=np.max(ADTtraining)
        UGOSmax=np.max(UGOStraining)
        VGOSmax=np.max(VGOStraining)

        Tmax=np.max(Ttraining)
        Smax=np.max(Straining)
        CHLmax=np.max(np.log10(CHLtraining))
        DSmax=np.max(DStraining)

        minMaxDict = {
                'LATmin' : LATmin,
                'LONmin' : LONmin,
                'SSTmin' : SSTmin,
                'SSSmin' : SSSmin,
                'SSDmin' : SSDmin,
                'OWPmin' : OWPmin,
                'ADTmin' : ADTmin,
                'UGOSmin' : UGOSmin,
                'VGOSmin' : VGOSmin,
                'Tmin' : Tmin,
                'Smin' : Smin,
                'CHLmin' : CHLmin,
                'DSmin' : DSmin,
                'LATmax' : LATmax,
                'LONmax' : LONmax,
                'SSTmax' : SSTmax,
                'SSSmax' : SSSmax,
                'SSDmax' : SSDmax,
                'OWPmax' : OWPmax,
                'ADTmax' : ADTmax,
                'UGOSmax' : UGOSmax,
                'VGOSmax' : VGOSmax,
                'Tmax' : Tmax,
                'Smax' : Smax,
                'CHLmax' : CHLmax,
                'DSmax' : DSmax,
        }

        ####################################
        #PREPROCESS INPUT AND TARGET DATA FOR TRAINING
        ####################################

        JD1=np.cos(2*np.pi*(JDtraining/365)+1)
        JD2=np.sin(2*np.pi*(JDtraining/365)+1)

        X0=JD1
        X1=JD2
        X2=(LATtraining-LATmin)/(LATmax-LATmin)
        X3=(LONtraining-LONmin)/(LONmax-LONmin)
        X4=(SSTtraining-SSTmin)/(SSTmax-SSTmin)
        X5=(SSStraining-SSSmin)/(SSSmax-SSSmin)
        X6=(np.log10(OWPtraining)-OWPmin)/(OWPmax-OWPmin)
        X7=(ADTtraining-ADTmin)/(ADTmax-ADTmin)
        X8=(UGOStraining-UGOSmin)/(UGOSmax-UGOSmin)
        X9=(VGOStraining-VGOSmin)/(VGOSmax-VGOSmin)
        X10=(SSDtraining-SSDmin)/(SSDmax-SSDmin)

        Y0=(Ttraining-Tmin)/(Tmax-Tmin)
        Y1=(np.log10(CHLtraining)-CHLmin)/(CHLmax-CHLmin)

        ####################################

        ####################################
        #PREPROCESS INPUT AND TARGET DATA FOR TEST
        ####################################

        JD1_test=np.cos(2*np.pi*(JDtest/365)+1)
        JD2_test=np.sin(2*np.pi*(JDtest/365)+1)

        X0_test=JD1_test
        X1_test=JD2_test
        X2_test=(LATtest-LATmin)/(LATmax-LATmin)
        X3_test=(LONtest-LONmin)/(LONmax-LONmin)
        X4_test=(SSTtest-SSTmin)/(SSTmax-SSTmin)
        X5_test=(SSStest-SSSmin)/(SSSmax-SSSmin)
        X6_test=(np.log10(OWPtest)-OWPmin)/(OWPmax-OWPmin)
        X7_test=(ADTtest-ADTmin)/(ADTmax-ADTmin)
        X8_test=(UGOStest-UGOSmin)/(UGOSmax-UGOSmin)
        X9_test=(VGOStest-VGOSmin)/(VGOSmax-VGOSmin)
        X10_test=(SSDtest-SSDmin)/(SSDmax-SSDmin)

        Y0_test=(Ttest-Tmin)/(Tmax-Tmin)
        Y1_test=(np.log10(CHLtest)-CHLmin)/(CHLmax-CHLmin)


        n_depth = X0.shape[1]
        n_samples= X0.shape[0]
        n_steps_out = 1#fixed
        n_var_in=11
        n_var_out=2

        #prepare data for training input

        X = np.zeros((n_samples, n_depth, n_var_in))

        # for i_var in range(n_var_in):
        #         cmd = 'X[:,i_var]=X' + str(i_var)+'[:,0]'
        # exec(cmd)

        X[:, :, 0] = X0[:, :]
        X[:, :, 1] = X1[:, :]
        X[:, :, 2] = X2[:, :]
        X[:, :, 3] = X3[:, :]
        X[:, :, 4] = X4[:, :]
        X[:, :, 5] = X5[:, :]
        X[:, :, 6] = X6[:, :]
        X[:, :, 7] = X7[:, :]
        X[:, :, 8] = X8[:, :]
        X[:, :, 9] = X9[:, :]
        X[:, :, 10] = X10[:, :]

        Y = np.stack([Y0, Y1], axis = -1)

        # cmd_str='Y0'
        # for i_var in range(1,n_var_out):
        #         cmd_str=cmd_str+',Y'+str(i_var)
        # cmd='Y=hstack(('+cmd_str+'))'
        # exec(cmd)


        n_var_in_test=11
        n_var_out_test=2

        
        n_samples_test= X0_test.shape[0]

        X_test=np.zeros((n_samples_test, n_depth, n_var_in_test))

        X_test[:, :, 0] = X0_test
        X_test[:, :, 1] = X1_test
        X_test[:, :, 2] = X2_test
        X_test[:, :, 3] = X3_test
        X_test[:, :, 4] = X4_test
        X_test[:, :, 5] = X5_test
        X_test[:, :, 6] = X6_test
        X_test[:, :, 7] = X7_test
        X_test[:, :, 8] = X8_test
        X_test[:, :, 9] = X9_test
        X_test[:, :, 10] = X10_test

        Y_test = np.stack([Y0_test, Y1_test], axis = -1)
     
        # X_test=np.zeros((n_samples_test,n_var_in_test))
        # for i_var_test in range(n_var_in_test):
        #         cmd = 'X_test[:,i_var_test]=X' + str(i_var_test)+'_test[:,0]'
        #         exec(cmd)

        # cmd_str='Y0_test'
        # for i_var_test in range(1,n_var_out_test):
        #         cmd_str=cmd_str+',Y'+str(i_var_test)+'_test'
        # cmd='Y_test=hstack(('+cmd_str+'))'
        # exec(cmd)
        #stop

        return X, Y, X_test, Y_test, minMaxDict
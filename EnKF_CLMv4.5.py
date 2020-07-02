#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
- Implementation of Ensemble Kalman Filter for CLMv4.5
- Uses parameter perturbation, covariance localization
- This is for doing regional run (25km gridcells) but can be modified for single column run as well
'''
import numpy as np
import os
import glob
import subprocess
import netCDF4 as nc
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import h5py as hdf

np.random.seed(9001)

#----------
# Case set-up part
# Need to run this only once, while setting up the case directory
#----------

# Creating the $CASEROOT directory
CASEROOT_DIR = '/home/ngg/shared/CESM1/cesm1_2_2/scripts'
os.chdir(CASEROOT_DIR)

# NOTE:: no '/' in the end
CASE_NAME = "mn_25km_t501_enkf_21_allcorecns_16yr"
CASE_DIR = '/home/ngg/anura003/calib_tests/regional_tests/mn_25km/CORRECTED_test_runs/final_runs2/%s' % CASE_NAME
os.mkdir('/scratch.global/anura003/calib_tests/regional_tests/mn_25km/final_runs/draietc/%s' % CASE_NAME)
os.mkdir('/scratch.global/anura003/calib_tests/regional_tests/mn_25km/final_runs/draietc/%s/rundir' % CASE_NAME)
RUNDIR = '/home/ngg/anura003/calib_tests/regional_tests/mn_25km/CORRECTED_test_runs/final_runs2/%s/rundir' % CASE_NAME

COMPSET = 'ICRUCLM45'

subprocess.check_output(['./create_newcase', '-case=%s' % CASE_DIR, '-compset=%s' % COMPSET, '-mach=mesabi', '-res=CLM_USRDAT'])

os.chdir(CASE_DIR)

#----------
# Change CLM's env_run.xml file parameter
#----------

grid_name = "mn_25km"
N = 100
total_processor = 24

subprocess.check_output(['./xmlchange', 'STOP_OPTION=nyears'])
subprocess.check_output(['./xmlchange', 'STOP_N=1'])
subprocess.check_output(['./xmlchange', 'DIN_LOC_ROOT_CLMFORC=/home/ngg/shared/cesm_inputdata/edge_clipped_atm_forcing.daymet_narr'])
subprocess.check_output(['./xmlchange', 'DATM_MODE=CLM1PT'])
subprocess.check_output(['./xmlchange', 'CLM_USRDAT_NAME=%s' % grid_name])
subprocess.check_output(['./xmlchange', 'CLM_FORCE_COLDSTART=off'])
subprocess.check_output(['./xmlchange', 'ATM_DOMAIN_FILE=mod_domain_25km_compgrid.nc'])
subprocess.check_output(['./xmlchange', 'ATM_DOMAIN_PATH=/home/ngg/anura003/domain_files'])
subprocess.check_output(['./xmlchange', 'LND_DOMAIN_FILE=mod_domain_25km_compgrid.nc'])
subprocess.check_output(['./xmlchange', 'LND_DOMAIN_PATH=/home/ngg/anura003/domain_files'])
subprocess.check_output(['./xmlchange', 'NINST_LND=%i' % N])
subprocess.check_output(['./xmlchange', 'NTASKS_LND=650'])
subprocess.check_output(['./xmlchange', 'CONTINUE_RUN=FALSE'])
subprocess.check_output(['./xmlchange', 'RUN_STARTDATE=2000-01-01'])
subprocess.check_output(['./xmlchange', 'DATM_CLMNCEP_YR_ALIGN=2000'])
subprocess.check_output(['./xmlchange', 'DATM_CLMNCEP_YR_START=2000'])
subprocess.check_output(['./xmlchange', 'CLM_BLDNML_OPTS=-mask USGS'])

subprocess.check_output(['./xmlchange', 'REST_OPTION=$STOP_OPTION'])
subprocess.check_output(['./xmlchange', 'REST_N=$STOP_N'])
subprocess.check_output(['./xmlchange', 'DATM_CLMNCEP_YR_END=2015'])

subprocess.check_output(['./xmlchange', 'RUNDIR=%s' % RUNDIR])

#----------
# This section is for setting up the user_nl_clm_00* files in CLM for the ensemble runs
# Make the rundir directory, and copy the relevant restart files into it
#----------

os.chdir(CASE_DIR)
os.mkdir('rundir')
# creating the directory where surface data files will be stored and updated.
# REMEMBER to copy the surface data files before the start of model run
os.mkdir('surface_data')

# directory which will contain surface data
surfdata_loc = CASE_DIR + '/surface_data'

# copying restart and other relevant files to rundir folder (needed for continuing previous run)
for i in range(1, N + 1):
    mc_number = str(i).zfill(4)
    filename = 'user_nl_clm_' + mc_number
    f = open(filename, 'w')
    f.write("hist_nhtfrq = -8760\nfsurdat = '%s/surfdata_MN_25km_pol19_glassClim_%s.nc'\nfinidat = '%s/mn_25km_t501_enkf_13_draietc_allcorecns_16yr.clm2_%s.r.2016-01-01-00000.nc'" %
            (surfdata_loc, str(i).zfill(3), RUNDIR, mc_number))
    shutil.copy2('/scratch.global/anura003/calib_tests/regional_tests/mn_25km/final_runs2/cheyenne_mn_25km_t501_enkf_13_draietc_allcorecns_16yr/mn_25km_t501_enkf_13_draietc_allcorecns_16yr.clm2_%s.rh0.2016-01-01-00000.nc' %
                 mc_number, '%s' % RUNDIR)
    shutil.copy2('/scratch.global/anura003/calib_tests/regional_tests/mn_25km/final_runs2/cheyenne_mn_25km_t501_enkf_13_draietc_allcorecns_16yr/mn_25km_t501_enkf_13_draietc_allcorecns_16yr.clm2_%s.r.2016-01-01-00000.nc' %
                 mc_number, '%s' % RUNDIR)
    shutil.copy2('/scratch.global/anura003/calib_tests/regional_tests/mn_25km/final_runs2/cheyenne_mn_25km_t501_enkf_13_draietc_allcorecns_16yr/rpointer.lnd_%s' %
                 mc_number, '%s' % RUNDIR)
    f.close()

shutil.copy2('/scratch.global/anura003/calib_tests/regional_tests/mn_25km/final_runs2/cheyenne_mn_25km_t501_enkf_13_draietc_allcorecns_16yr/rpointer.atm',
             '%s' % RUNDIR)
shutil.copy2('/scratch.global/anura003/calib_tests/regional_tests/mn_25km/final_runs2/cheyenne_mn_25km_t501_enkf_13_draietc_allcorecns_16yr/rpointer.drv',
             '%s' % RUNDIR)
shutil.copy2('/scratch.global/anura003/calib_tests/regional_tests/mn_25km/final_runs2/cheyenne_mn_25km_t501_enkf_13_draietc_allcorecns_16yr/mn_25km_t501_enkf_13_draietc_allcorecns_16yr.datm.rs1.2016-01-01-00000.bin', '%s' % RUNDIR)
shutil.copy2('/scratch.global/anura003/calib_tests/regional_tests/mn_25km/final_runs2/cheyenne_mn_25km_t501_enkf_13_draietc_allcorecns_16yr/mn_25km_t501_enkf_13_draietc_allcorecns_16yr.cpl.r.2016-01-01-00000.nc', '%s' % RUNDIR)

# copying the surface data files
orig_surfd_files = '/scratch.global/anura003/calib_tests/regional_tests/mn_25km/final_runs2/cheyenne_mn_25km_t501_enkf_13_draietc_allcorecns_16yr/surface_data'

temp_files = sorted(glob.glob('%s/*.nc' % orig_surfd_files))[0:N]
for i, f in enumerate(temp_files):
    shutil.copy2(f, surfdata_loc)

# user_nl_datm and stream files
for i, f in enumerate(glob.glob('/home/ngg/anura003/domain_files/extra_files_py_setup/state_25km/mod_stream_files/user_datm.streams.txt.*')):
    shutil.copy2(f, CASE_DIR + '/')
    shutil.copy2(f, RUNDIR + '/')
shutil.copy2(
    '/home/ngg/anura003/domain_files/extra_files_py_setup/state_25km/mod_stream_files/user_nl_datm', CASE_DIR + '/')

# changing the file permission for user_stream files (CLM requires this)
for i, f in enumerate(glob.glob('user_datm.streams.txt.*')):
    os.chmod(f, 0o644)
for i, f in enumerate(glob.glob('%s/user_datm.streams.txt.*' % RUNDIR)):
    os.chmod(f, 0o644)

# copy the source code modifications in src.clm folder
for i, f in enumerate(glob.glob('/home/ngg/anura003/domain_files/extra_files_py_setup/new_additions/src.hydP_fdrai_errCheck_QCcorrection_etc_new/*')):
    dest = CASE_DIR + '/SourceMods/src.clm/'
    shutil.copy2(f, dest)
    # overwriting the lai read file
    shutil.copy2(
        '/home/ngg/anura003/regional_tests_files/sfd_files/mn_25km_GLASS_LAI_trans/clipped_mn_25km_GLASS_LAI_trans/STATICEcosysDynMod.F90', dest)

# preparing the observations
obs_df = pd.read_excel('/home/ngg/anura003/regional_tests_files/obs_dataframes/mn25km_final_corrected_obs_aggregated_16yr.xlsx', index_col=0)
total_time = 59

start_day = pd.to_datetime('2000-01-01')
# pd.offset to convert timedelta object to number of days
init_interval = int((obs_df['Read Time/Date (CST)'][0] - start_day) / pd.offsets.Day(1))  # days for the first obs
# CLM doesn't consider leap years - adjustments for that
if obs_df['Read Time/Date (CST)'][0] > pd.to_datetime('2012-02-28'):
    init_interval -= 4
elif obs_df['Read Time/Date (CST)'][0] > pd.to_datetime('2008-02-28'):
    init_interval -= 3
elif obs_df['Read Time/Date (CST)'][0] > pd.to_datetime('2004-02-28'):
    init_interval -= 2
elif obs_df['Read Time/Date (CST)'][0] > pd.to_datetime('2000-02-28'):
    init_interval -= 1
subprocess.check_output(['./xmlchange', 'STOP_N=%i' % init_interval])


os.chdir(CASE_DIR)

A_ones = np.ones((N, N)) * 1 / N

nlat = 24
nlon = 29
size_grid = nlat * nlon
state_vars = 65
size_state = state_vars * nlat * nlon

# # LOCALIZATION - HARD CUT
# from scipy.linalg import block_diag
# temp_loc = np.ones((66, 66))
# corr_matrix = block_diag(*([temp_loc] * 25))


# LOCALIZATION - TAPERING FUNCTION (GASPARI-COHN)
# Gaspari-Cohn tapering function for localization
# covariance localization
def corr_gaspari_cohn():
    i = np.where((0 <= dist_matrix) & (dist_matrix <= Fc))
    corr_matrix[i] = -0.25 * (dist_matrix[i] / Fc)**5 + 0.5 * (dist_matrix[i] / Fc)**4 + 0.625 * (dist_matrix[i] / Fc)**3 - (5. / 3.) * (dist_matrix[i] / Fc)**2 + 1
    print ('done with 1')
    i = np.where((Fc < dist_matrix) & (dist_matrix <= 2 * Fc))
    corr_matrix[i] = (1. / 12.) * (dist_matrix[i] / Fc)**5 - 0.5 * (dist_matrix[i] / Fc)**4 + 0.625 * (dist_matrix[i] /
                                                                                                       Fc)**3 + (5. / 3.) * (dist_matrix[i] / Fc)**2 - 5 * (dist_matrix[i] / Fc) + 4 - (2. / 3.) * (dist_matrix[i] / Fc)**-1
    print ('done with 2')
    i = np.where(dist_matrix > 2 * Fc)
    corr_matrix[i] = 0
    print ('done with 3')


def dist_finder_euc(obs):
    D_ij = np.zeros((nlat, nlon))
    for i in range(nlat):
        for j in range(nlon):
            temp = np.asarray([i, j])
            D_ij[i, j] = np.linalg.norm(temp - obs)
    return (D_ij.flatten())

# first calculate the distance matrix
dist_matrix = np.zeros((size_state, size_state))
k = 0
for a in range(nlat):
    for b in range(nlon):
        dist = dist_finder_euc(np.asarray([a, b])) * 25
        dist_matrix[k:k + state_vars, :] = np.repeat(dist.T, state_vars)
        k += state_vars
        print (k)

# distance matrix is used in the Gaspari-Cohn calculation
# calculate the corr matrix
lc = 25 * 1  #tapering distance
Fc = np.sqrt(10 / 3) * lc
corr_matrix = np.zeros((size_state, size_state))
corr_gaspari_cohn()
del dist_matrix #to free up memory (not a good practice :(; think about this later)


os.chdir(CASE_DIR)

# CLM Set-up, build
pipe = subprocess.Popen(['./cesm_setup'], stdout=subprocess.PIPE)
result = pipe.communicate()[0]
print (result)

pipe = subprocess.Popen(['./%s.build' % CASE_NAME], stdout=subprocess.PIPE)
result = pipe.communicate()[0]
print (result)

#------------
# Run the model, EnKF analysis & update
#------------

os.chdir(CASE_DIR)
total_obs = nlat * nlon
total_iter = 1
# creating an ordered list of all the surface data files
surfd_files_list = sorted(glob.glob('%s/surface_data/*.nc' % CASE_DIR))

# arrays for recording prediction and updated values
# Variables being updated (CLM's Variable code):
# - Water Table Depth (ZWT)
# - Slope of soil water retention curve (BSW); Saturated hydraulic conductivity (HKSAT); Saturated soil matric potential (SUCSAT); Saturated Soil water content(WATSAT)
# - Subsurface Drainage Parameters (f_drai, q_drai_max)
# - Surface Runoff Parameters (f_over, FMAX)

#creating HDF files for storing the variables
#HDF was most memory efficient

#PREDICTION
pred_record = hdf.File(
    '/home/ngg/anura003/calib_tests/enkf_nparray_outputs/regional_tests/mn_25km/CORRECTED_test_runs/final_runs2/%s_pred_record.h5' % CASE_NAME, 'w')
pred_wt = pred_record.create_dataset('pred_wt', shape=(
    total_iter, total_obs, total_time, N), dtype=np.float32)
pred_bsw = pred_record.create_dataset('pred_bsw', shape=(
    total_iter, 15, total_obs, total_time, N), dtype=np.float32)
pred_hksat = pred_record.create_dataset('pred_hksat', shape=(
    total_iter, 15, total_obs, total_time, N), dtype=np.float32)
pred_sucsat = pred_record.create_dataset('pred_sucsat', shape=(
    total_iter, 15, total_obs, total_time, N), dtype=np.float32)
pred_watsat = pred_record.create_dataset('pred_watsat', shape=(
    total_iter, 15, total_obs, total_time, N), dtype=np.float32)
pred_f_drai = pred_record.create_dataset('pred_f_drai', shape=(
    total_iter, total_obs, total_time, N), dtype=np.float32)
pred_q_drai_max = pred_record.create_dataset('pred_q_drai_max', shape=(
    total_iter, total_obs, total_time, N), dtype=np.float32)
pred_fover = pred_record.create_dataset('pred_fover', shape=(
    total_iter, total_obs, total_time, N), dtype=np.float32)
pred_fmax = pred_record.create_dataset('pred_fmax', shape=(
    total_iter, total_obs, total_time, N), dtype=np.float32)

#UPDATES
upd_record = hdf.File(
    '/home/ngg/anura003/calib_tests/enkf_nparray_outputs/regional_tests/mn_25km/CORRECTED_test_runs/final_runs2/%s_upd_record.h5' % CASE_NAME, 'w')
upd_wt = upd_record.create_dataset('upd_wt', shape=(
    total_iter, total_obs, total_time, N), dtype=np.float32)
upd_bsw = upd_record.create_dataset('upd_bsw', shape=(
    total_iter, 15, total_obs, total_time, N), dtype=np.float32)
upd_hksat = upd_record.create_dataset('upd_hksat', shape=(
    total_iter, 15, total_obs, total_time, N), dtype=np.float32)
upd_sucsat = upd_record.create_dataset('upd_sucsat', shape=(
    total_iter, 15, total_obs, total_time, N), dtype=np.float32)
upd_watsat = upd_record.create_dataset('upd_watsat', shape=(
    total_iter, 15, total_obs, total_time, N), dtype=np.float32)
upd_f_drai = upd_record.create_dataset('upd_f_drai', shape=(
    total_iter, total_obs, total_time, N), dtype=np.float32)
upd_q_drai_max = upd_record.create_dataset('upd_q_drai_max', shape=(
    total_iter, total_obs, total_time, N), dtype=np.float32)
upd_fover = upd_record.create_dataset('upd_fover', shape=(
    total_iter, total_obs, total_time, N), dtype=np.float32)
upd_fmax = upd_record.create_dataset('upd_fmax', shape=(
    total_iter, total_obs, total_time, N), dtype=np.float32)

#option to store the covariance matrices if doing a smaller regional run
# P_record_fcast = np.zeros((total_iter, size_state, size_state, total_time))
# P_record_analysis = np.zeros((total_iter, size_state, size_state, total_time))
# #R_record = np.zeros((total_iter, total_obs, total_obs, total_time))

# P_record_fcast_afterLoc = np.zeros(
#     (total_iter, size_state, size_state, total_time))

obs_record = np.zeros((total_obs, total_time))

# option to store parameter perturbations when doing smaller regional runs
# pre_pert_theta_record = np.zeros((total_iter, 61, total_obs, N))
# post_pert_theta_record = np.zeros((total_iter, 61, total_obs, N))

leap_yr = [2000, 2004, 2008, 2012]
total_df_length = 59 #same as total time
iter_no = 0 #counting iterations

while iter_no < total_iter:
    subprocess.check_output(['./xmlchange', 'STOP_OPTION=ndays'])
    subprocess.check_output(['./xmlchange', 'STOP_N=%i' % init_interval])

    for obs_no in range(total_time):
        # run the model -- prediction step
        pipe = subprocess.Popen(['./%s.run' % CASE_NAME], stdout=subprocess.PIPE)
        result = pipe.communicate()[0]
        print (result)

        os.chdir(RUNDIR)

        # For generating the date part of the restart file : .r files are one month (or one output timestep) ahead of the model run date (output=2000-01 :: .r=2000-02)
        temp_r_list = sorted(glob.glob('%s.clm2_0001.r.*.nc' % CASE_NAME))
        rfile_date = temp_r_list[-1][-19:-3]

        # extracting out yr to check for leap yr later
        rfile_yr = int(rfile_date[0:4])

        # list of all the .r files corresponding to the model run
        rfiles_list = sorted(glob.glob('*clm2*.r.*%s*' % rfile_date))

        # forming the state vector
        A = np.zeros((size_state, N))
        for idx, files in enumerate(rfiles_list):
            k = 0
            r_file = nc.Dataset(files)
            no_gcell = len(r_file['grid1d_lon'])
            no_cols = len(r_file['ZWT'])
            temp = np.zeros((no_cols, 5))
            temp[:, 0] = r_file['cols1d_wtxy'][:]  # weight of each col
            temp[:, 1] = r_file['cols1d_ixy'][:]  # i/long of each col
            temp[:, 2] = r_file['cols1d_jxy'][:]  # j/lat of each col
            temp[:, 3] = r_file['ZWT'][:]
            temp[:, 4] = range(no_cols)

            sorted_by_wt = temp[temp[:, 0].argsort()]
            relevant_cells = np.flip(sorted_by_wt, 0)[0:no_gcell, :]
            idx_sorted = np.lexsort((relevant_cells[:, 1], relevant_cells[:, 2]))
            order_by_idx = relevant_cells[idx_sorted]

            for cell_id in range(size_grid):
                A[k:k + 1, idx] = order_by_idx[cell_id, 3:4]
                k += state_vars
            r_file.close()

        # getting parameter values
        for idx, files in enumerate(surfd_files_list):
            temp = nc.Dataset(files)

            k = 1  # drai start
            for nj in range(nlat):
                for ni in range(nlon):
                    A[k, idx] = temp['f_drai'][nj, ni]
                    k += state_vars

            k = 2  # bsw start
            for nj in range(nlat):
                for ni in range(nlon):
                    A[k:k + 15, idx] = temp['BSW'][:, nj, ni]
                    k += state_vars

            k = 17
            for nj in range(nlat):
                for ni in range(nlon):
                    A[k:k + 15, idx] = np.log(temp['HKSAT'][:, nj, ni])
                    k += state_vars

            k = 32
            for nj in range(nlat):
                for ni in range(nlon):
                    A[k:k + 15, idx] = temp['SUCSAT'][:, nj, ni]
                    k += state_vars

            k = 47
            for nj in range(nlat):
                for ni in range(nlon):
                    A[k:k + 15, idx] = temp['WATSAT'][:, nj, ni]
                    k += state_vars

            k = 62  # q_drai_max
            for nj in range(nlat):
                for ni in range(nlon):
                    A[k, idx] = temp['q_drai_max'][nj, ni]
                    k += state_vars

            k = 63  # fover
            for nj in range(nlat):
                for ni in range(nlon):
                    A[k, idx] = temp['fover'][nj, ni]
                    k += state_vars

            k = 64  # fmax
            for nj in range(nlat):
                for ni in range(nlon):
                    A[k, idx] = temp['FMAX'][nj, ni]
                    k += state_vars

            temp.close()

        # recording prediction values
        #'pred' values of hyd param already copied above
        pred_wt[iter_no, :, obs_no, :] = A[0::state_vars, :]
        pred_f_drai[iter_no, :, obs_no, :] = A[1::state_vars, :]
        k = 2
        for i in range(total_obs):
            pred_bsw[iter_no, :, i, obs_no, :] = A[k:k + 15, :]
            k += 15
            pred_hksat[iter_no, :, i, obs_no, :] = np.exp(A[k:k + 15, :])
            k += 15
            pred_sucsat[iter_no, :, i, obs_no, :] = A[k:k + 15, :]
            k += 15
            pred_watsat[iter_no, :, i, obs_no, :] = A[k:k + 15, :]
            k += 15 + 5
        pred_q_drai_max[iter_no, :, obs_no, :] = A[62::state_vars, :]
        pred_fover[iter_no, :, obs_no, :] = A[63::state_vars, :]
        pred_fmax[iter_no, :, obs_no, :] = A[64::state_vars, :]

        A_mean = np.dot(A, A_ones)
        A_ens_pert = A - A_mean
        P_ens = (np.dot(A_ens_pert, A_ens_pert.T)) / (N - 1)

        # COVARIANCE INFLATION
        P_ens = np.multiply(P_ens, 1.5)

        # Recording P_fcast
        # P_record_fcast[iter_no, :, :, obs_no] = P_ens

        # LOCALIZATION
        P_ens = np.multiply(corr_matrix, P_ens)
        # P_record_fcast_afterLoc[iter_no, :, :, obs_no] = P_ens

        ####
        # prepare the observations for the stop
        # extract the relevant info from the master obs database
        # note the column numbers being used - can vary and thus cause error
        ####
        temp_db_obs_well_id = obs_df.iloc[obs_no, 1::6]
        temp_db_obs_wtd = obs_df.iloc[obs_no, 2::6]
        temp_db_obs_j_lat = obs_df.iloc[obs_no, 4::6]
        temp_db_obs_i_lon = obs_df.iloc[obs_no, 3::6]

        # counting the number of observations for the assimilation step
        db_obs_count1 = np.count_nonzero(~pd.isnull(temp_db_obs_wtd))

        # stack all the series extracted above
        temp_db_obs_all = np.stack((temp_db_obs_well_id, temp_db_obs_j_lat, temp_db_obs_i_lon, temp_db_obs_wtd), axis=1)

        # keeping only the rows which has observations
        # CAREFUL WITH THIS -- POTENTIAL SOURCE OF ERROR
        obs_mask = np.all(~pd.isnull(temp_db_obs_all), axis=1)
        temp_db_obs_all = temp_db_obs_all[obs_mask]
        # if need to convert object dtype array to a float array
        temp_db_obs_all = np.vstack(temp_db_obs_all[:, :]).astype(np.float)
        temp_db_obs_all[:, 0:3] = temp_db_obs_all[:, 0:3].astype(int)

        obs_wt = temp_db_obs_all[:, 3] * 0.3048
        # prepare your observation perturbations
        d = np.array([obs_wt] * N).T
        epsilon = np.random.normal(0, 0.45, (np.size(temp_db_obs_all, 0), N))
        D = d + epsilon
        R_ens = (np.dot(epsilon, epsilon.T)) / (N - 1)

        # constructing H
        no_obs = db_obs_count1
        H = np.zeros((no_obs, size_state))
        for i in range(no_obs):
            cell_id = nlon * temp_db_obs_all[i, 1] + temp_db_obs_all[i, 2]
            H[i, int(state_vars * cell_id)] = 1

        # recording observation - only first iteration
        if iter_no == 0:
            for i in range(no_obs):
                cell_id = nlon * temp_db_obs_all[i, 1] + temp_db_obs_all[i, 2]
                obs_record[int(cell_id),
                           obs_no] = temp_db_obs_all[i, 3] * .3048

        # updating the state variable
        innovation = D - np.dot(H, A)
        HPH_t = np.dot(np.dot(H, P_ens), H.T)
        inv_element = np.linalg.inv(HPH_t + R_ens)
        K = np.dot(np.dot(P_ens, H.T), inv_element)
        A_analysis = A + np.dot(K, innovation)

        # checking and adjusting within reasonable limits
        A_analysis[0::state_vars, :] = np.clip(A_analysis[0::state_vars, :], 0.09, 70)  # WTD
        A_analysis[1::state_vars, :] = np.clip(A_analysis[1::state_vars, :], 0.1, 15)  # f_drai

        k = 2
        for i in range(nlat * nlon):
            A_analysis[k:k + 15,
                       :] = np.clip(A_analysis[k:k + 15, :], 0.1, 70)  # bsw
            k += 15
            # hksat
            tmp = np.exp(A_analysis[k:k + 15, :])
            A_analysis[k:k + 15, :] = np.log(np.clip(tmp, 10**-5, 9))
            k += 15
            # sucsat
            A_analysis[k:k + 15, :] = np.clip(A_analysis[k:k + 15, :], 2, 1100)
            k += 15
            # watsat
            A_analysis[k:k + 15,
                       :] = np.clip(A_analysis[k:k + 15, :], 0.1, 0.95)
            k += 15 + 5

        A_analysis[62::state_vars, :] = np.clip(
            A_analysis[62::state_vars, :], 10**-7, 10**-1)  # q_drai_max
        A_analysis[63::state_vars, :] = np.clip(
            A_analysis[63::state_vars, :], 0.1, 7)  # fover
        A_analysis[64::state_vars, :] = np.clip(
            A_analysis[64::state_vars, :], 0.01, 0.91)  # fmax

        # recording updated values
        upd_wt[iter_no, :, obs_no, :] = A_analysis[0::state_vars, :]
        upd_f_drai[iter_no, :, obs_no, :] = A_analysis[1::state_vars, :]
        k = 2
        for i in range(nlat * nlon):
            upd_bsw[iter_no, :, i, obs_no, :] = A_analysis[k:k + 15, :]
            k += 15
            upd_hksat[iter_no, :, i, obs_no, :] = np.exp(
                A_analysis[k:k + 15, :])
            k += 15
            upd_sucsat[iter_no, :, i, obs_no, :] = A_analysis[k:k + 15, :]
            k += 15
            upd_watsat[iter_no, :, i, obs_no, :] = A_analysis[k:k + 15, :]
            k += 15 + 5

        upd_q_drai_max[iter_no, :, obs_no, :] = A_analysis[62::state_vars, :]
        upd_fover[iter_no, :, obs_no, :] = A_analysis[63::state_vars, :]
        upd_fmax[iter_no, :, obs_no, :] = A_analysis[64::state_vars, :]

        # recording P and R covariance matrix
        # P_record_fcast[iter_no, :, :, obs_no] = P_ens
        #R_record[iter_no, :, :, obs_no] = R_ens #commenting this out because size of R_ens is not fixed

        # updated P
        # A_mean_analysis = np.dot(A_analysis, A_ones)
        # A_ens_pert_analysis = A_analysis - A_mean_analysis
        # P_ens_analysis = (np.dot(A_ens_pert_analysis,
        #                          A_ens_pert_analysis.T)) / (N - 1)

        # recording updated P
        # P_record_analysis[iter_no, :, :, obs_no] = P_ens_analysis

        # updating restart files
        for idx, files in enumerate(rfiles_list):
            k = 0
            temps_ds = nc.Dataset(files, 'r+')
            no_gcell = len(temps_ds['grid1d_lon'])
            no_cols = len(temps_ds['ZWT'])
            temp = np.zeros((no_cols, 4))
            temp[:, 0] = temps_ds['cols1d_wtxy'][:]  # weight of each col
            temp[:, 1] = temps_ds['cols1d_ixy'][:]  # i/long of each col
            temp[:, 2] = temps_ds['cols1d_jxy'][:]  # j/lat of each col
            temp[:, 3] = range(no_cols)  # col idx

            sorted_by_wt = temp[temp[:, 0].argsort()]
            relevant_cells = np.flip(sorted_by_wt, 0)[0:no_gcell, :]
            idx_sorted = np.lexsort(
                (relevant_cells[:, 1], relevant_cells[:, 2]))
            order_by_idx = relevant_cells[idx_sorted]
            l = [int(x) for x in order_by_idx[:, 3]]

            temps_ds['ZWT'][l] = A_analysis[0::state_vars, idx]
            temps_ds.close()
        
        #-------
        # parameter perturbation - for avoiding ensemble collapse
        # 64 - no of prms in one gridcell state vector
        #-------
        pre_pert_theta = np.zeros((nlat * nlon * 64, N))
        pre_pert_theta[0:nlat * nlon,
                       :] = A_analysis[1::state_vars, :]  # drainage
        j = size_grid  # 25
        k = 2
        for i in range(size_grid):
            pre_pert_theta[j:j + 15, :] = A_analysis[k:k + 15, :]
            j += 15
            k += state_vars

        j = size_grid * 16  # size_grid + size_grid*15 #400 - for 5*5 grid
        k = 17
        for i in range(size_grid):
            pre_pert_theta[j:j + 15, :] = A_analysis[k:k + 15, :]
            j += 15
            k += state_vars

        j = size_grid * 31  # size_grid + 2*size_grid*15 #775 - for 5*5 grid
        k = 32
        for i in range(size_grid):
            pre_pert_theta[j:j + 15, :] = A_analysis[k:k + 15, :]
            j += 15
            k += state_vars

        j = size_grid * 46  # size_grid + 3*size_grid*15 #1150
        k = 47
        for i in range(size_grid):
            pre_pert_theta[j:j + 15, :] = A_analysis[k:k + 15, :]
            j += 15
            k += state_vars

        pre_pert_theta[size_grid * 61:size_grid * 62,
                       :] = A_analysis[62::state_vars, :]  # q_drai_max
        pre_pert_theta[size_grid * 62:size_grid * 63,
                       :] = A_analysis[63::state_vars, :]  # fover
        pre_pert_theta[size_grid * 63:size_grid * 64,
                       :] = A_analysis[64::state_vars, :]  # fmax

        h = 0.3
        alpha = 0.95
        var_pre_pert_theta = np.var(pre_pert_theta, axis=1)
        T_var = h**2 * var_pre_pert_theta
        mean_pre_pert_theta = np.mean(pre_pert_theta, axis=1)
        post_pert_theta = np.zeros((len(pre_pert_theta[:, 0]), N))

        # fdrai
        post_pert_theta[0:size_grid, :] = (alpha * pre_pert_theta[0:size_grid, :]) + ((1 - alpha) * np.tile(
            mean_pre_pert_theta[0:size_grid], (N, 1)).T) + np.random.normal(0, np.sqrt(np.max(T_var[0:size_grid])), N)
        # bsw
        post_pert_theta[size_grid:size_grid * 16, :] = (alpha * pre_pert_theta[size_grid:size_grid * 16, :]) + ((1 - alpha) * np.tile(
            mean_pre_pert_theta[size_grid:size_grid * 16], (N, 1)).T) + np.random.normal(0, np.sqrt(np.max(T_var[size_grid:size_grid * 16])), N)
        # hksat
        post_pert_theta[size_grid * 16:size_grid * 31, :] = (alpha * pre_pert_theta[size_grid * 16:size_grid * 31, :]) + ((1 - alpha) * np.tile(
            mean_pre_pert_theta[size_grid * 16:size_grid * 31], (N, 1)).T) + np.random.normal(0, np.sqrt(np.max(T_var[size_grid * 16:size_grid * 31])), N)
        # sucsat
        post_pert_theta[size_grid * 31:size_grid * 46, :] = (alpha * pre_pert_theta[size_grid * 31:size_grid * 46, :]) + ((1 - alpha) * np.tile(
            mean_pre_pert_theta[size_grid * 31:size_grid * 46], (N, 1)).T) + np.random.normal(0, np.sqrt(np.max(T_var[size_grid * 31:size_grid * 46])), N)
        # watsat
        post_pert_theta[size_grid * 46:size_grid * 61, :] = (alpha * pre_pert_theta[size_grid * 46:size_grid * 61, :]) + ((1 - alpha) * np.tile(
            mean_pre_pert_theta[size_grid * 46:size_grid * 61], (N, 1)).T) + np.random.normal(0, np.sqrt(np.max(T_var[size_grid * 46:size_grid * 61])), N)
        # qdrai_max
        post_pert_theta[size_grid * 61:size_grid * 62, :] = (alpha * pre_pert_theta[size_grid * 61:size_grid * 62, :]) + ((1 - alpha) * np.tile(
            mean_pre_pert_theta[size_grid * 61:size_grid * 62], (N, 1)).T) + np.random.normal(0, np.sqrt(np.max(T_var[size_grid * 61:size_grid * 62])), N)
        # fover
        post_pert_theta[size_grid * 62:size_grid * 63, :] = (alpha * pre_pert_theta[size_grid * 62:size_grid * 63, :]) + ((1 - alpha) * np.tile(
            mean_pre_pert_theta[size_grid * 62:size_grid * 63], (N, 1)).T) + np.random.normal(0, np.sqrt(np.max(T_var[size_grid * 62:size_grid * 63])), N)
        # fmax
        post_pert_theta[size_grid * 63:size_grid * 64, :] = (alpha * pre_pert_theta[size_grid * 63:size_grid * 64, :]) + ((1 - alpha) * np.tile(
            mean_pre_pert_theta[size_grid * 63:size_grid * 64], (N, 1)).T) + np.random.normal(0, np.sqrt(np.max(T_var[size_grid * 63:size_grid * 64])), N)

        # checking if the perturbed values are reasonable
        post_pert_theta[0:size_grid, :] = np.clip(
            post_pert_theta[0:size_grid, :], 0.1, 15)  # f_drai
        post_pert_theta[size_grid:size_grid * 16, :] = np.clip(
            post_pert_theta[size_grid:size_grid * 16, :], 0.1, 70)  # bsw
        post_pert_theta[size_grid * 16:size_grid * 31, :] = np.log(np.clip(
            np.exp(post_pert_theta[size_grid * 16:size_grid * 31, :]), 10**-5, 9))  # hksat
        post_pert_theta[size_grid * 31:size_grid * 46, :] = np.clip(
            post_pert_theta[size_grid * 31:size_grid * 46, :], 2, 1100)  # sucsat
        post_pert_theta[size_grid * 46:size_grid * 61, :] = np.clip(
            post_pert_theta[size_grid * 46:size_grid * 61, :], 0.1, 0.95)  # watsat
        post_pert_theta[size_grid * 61:size_grid * 62, :] = np.clip(
            post_pert_theta[size_grid * 61:size_grid * 62, :], 10**-7, 10**-1)  # qdrai_max
        post_pert_theta[size_grid * 62:size_grid * 63, :] = np.clip(
            post_pert_theta[size_grid * 62:size_grid * 63, :], 0.1, 7)  # fover
        post_pert_theta[size_grid * 63:size_grid * 64, :] = np.clip(
            post_pert_theta[size_grid * 63:size_grid * 64, :], 0.01, 91)  # fmax

        # recording the values for debugging

        # recording the values for debugging
        # pre_pert_theta_record[iter_no, :, obs_no, :] = pre_pert_theta
        # post_pert_theta_record[iter_no, :, obs_no, :] = post_pert_theta

        # perturbed prm values in the analysis vector
        A_analysis[1::state_vars, :] = post_pert_theta[0:size_grid, :]
        j = size_grid
        k = 2
        for i in range(size_grid):
            A_analysis[k:k + 15, :] = post_pert_theta[j:j + 15, :]
            j += 15
            k += state_vars

        j = size_grid * 16  # size_grid + size_grid*15 #400 - for 5*5 grid
        k = 17
        for i in range(size_grid):
            A_analysis[k:k + 15, :] = post_pert_theta[j:j + 15, :]
            j += 15
            k += state_vars

        j = size_grid * 31  # size_grid + 2*size_grid*15 #775 - for 5*5 grid
        k = 32
        for i in range(size_grid):
            A_analysis[k:k + 15, :] = post_pert_theta[j:j + 15, :]
            j += 15
            k += state_vars

        j = size_grid * 46  # size_grid + 3*size_grid*15 #1150
        k = 47
        for i in range(size_grid):
            A_analysis[k:k + 15, :] = post_pert_theta[j:j + 15, :]
            j += 15
            k += state_vars

        A_analysis[62::state_vars, :] = post_pert_theta[size_grid *
                                                        61:size_grid * 62, :]  # qdrai_max
        A_analysis[63::state_vars, :] = post_pert_theta[size_grid *
                                                        62:size_grid * 63, :]  # fover
        A_analysis[64::state_vars, :] = post_pert_theta[size_grid *
                                                        63:size_grid * 64, :]  # fmax

        # updating surface data files
        for idx, files in enumerate(surfd_files_list):
            # Have not added any extra model noise to these for now
            temp = nc.Dataset(files, 'r+')

            k = 1  # drai start
            for nj in range(nlat):
                for ni in range(nlon):
                    temp['f_drai'][nj, ni] = A_analysis[k, idx]
                    k += state_vars

            k = 2  # bsw start
            for nj in range(nlat):
                for ni in range(nlon):
                    temp['BSW'][:, nj, ni] = A_analysis[k:k + 15, idx]
                    k += state_vars

            k = 17
            for nj in range(nlat):
                for ni in range(nlon):
                    temp['HKSAT'][:, nj, ni] = np.exp(
                        A_analysis[k:k + 15, idx])
                    k += state_vars

            k = 32
            for nj in range(nlat):
                for ni in range(nlon):
                    temp['SUCSAT'][:, nj, ni] = A_analysis[k:k + 15, idx]
                    k += state_vars

            k = 47
            for nj in range(nlat):
                for ni in range(nlon):
                    temp['WATSAT'][:, nj, ni] = A_analysis[k:k + 15, idx]
                    k += state_vars

            k = 62  # qdrai_max
            for nj in range(nlat):
                for ni in range(nlon):
                    temp['q_drai_max'][nj, ni] = A_analysis[k, idx]
                    k += state_vars

            k = 63  # fover
            for nj in range(nlat):
                for ni in range(nlon):
                    temp['fover'][nj, ni] = A_analysis[k, idx]
                    k += state_vars

            k = 64  # drai start
            for nj in range(nlat):
                for ni in range(nlon):
                    temp['FMAX'][nj, ni] = A_analysis[k, idx]
                    k += state_vars

            temp.close()

        # checking if it is the last iteration of the loop
        # if last iteration, don't calculate interval
        if obs_no != total_df_length - 1:
            os.chdir(CASE_DIR)
            interval = int((obs_df['Read Time/Date (CST)'][obs_no + 1] -
                            obs_df['Read Time/Date (CST)'][obs_no]) / pd.offsets.Day(1))

            present_obs = obs_df['Read Time/Date (CST)'][obs_no]
            next_obs = obs_df['Read Time/Date (CST)'][obs_no + 1]

            temp_leap = 0
            if next_obs > pd.to_datetime('2012-02-28') > present_obs:
                temp_leap += 1
            if next_obs > pd.to_datetime('2008-02-28') > present_obs:
                temp_leap += 1
            if next_obs > pd.to_datetime('2004-02-28') > present_obs:
                temp_leap += 1
            if next_obs > pd.to_datetime('2000-02-28') > present_obs:
                temp_leap += 1

            interval -= temp_leap
            interval = np.clip(interval, 1, None)  # min 1 day
            subprocess.check_output(['./xmlchange', 'STOP_N=%i' % interval])
            subprocess.check_output(['./xmlchange', 'CONTINUE_RUN=TRUE'])

        os.chdir(CASE_DIR)
        subprocess.check_output(['./xmlchange', 'CONTINUE_RUN=TRUE'])

    # after the last observation is assimilated, run the model till the last day
    os.chdir(CASE_DIR)
    end_day = pd.to_datetime('2004-12-31')
    # adding 1 to the interval so that the run is completed for the entire year
    # Otherwise, the rfile is generated for *-12-31 (i.e. the last day of the year)
    interval = int((end_day - obs_df['Read Time/Date (CST)']
                    [total_df_length - 1]) / pd.offsets.Day(1)) + 1
    interval = np.clip(interval, 1, None)
    subprocess.check_output(['./xmlchange', 'STOP_N=%i' % interval])
    subprocess.check_output(['./xmlchange', 'CONTINUE_RUN=TRUE'])

    # run the model -- last interval
    pipe = subprocess.Popen(['./%s.run' % CASE_NAME], stdout=subprocess.PIPE)
    result = pipe.communicate()[0]
    print (result)

    if iter_no != total_iter - 1:
        iter_no += 1
    else:
        iter_no += 1


pred_record.close()
upd_record.close()
np.save('/home/ngg/anura003/calib_tests/enkf_nparray_outputs/regional_tests/mn_25km/CORRECTED_test_runs/final_runs2/%s_obs_wt' %
        CASE_NAME, obs_record)

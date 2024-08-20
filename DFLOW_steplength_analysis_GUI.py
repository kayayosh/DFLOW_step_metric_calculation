#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:31:26 2024

@author: kayayoshida
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:14:13 2022

@author: kayayoshida
"""


import streamlit as st
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default='browser'
from scipy.signal import butter, filtfilt
import os
import matplotlib.pyplot as plt


savefile = True
# savefile = False
showfig = False
# showfig= True

st.title('Analyze data from DFLOW and calculate step metrics')

st.sidebar.subheader("Select a file or folder")

# Sidebar header
st.sidebar.header("Input Parameters")

# Step threshold input
stepthresh = st.sidebar.number_input(
    "Minimum step length for detection (stepthresh)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,  # Default value
    format="%.2f"
)

# Force threshold input
Fthresh = st.sidebar.number_input(
    "Percentage of BW to detect double support (Fthresh)",
    min_value=0.0,
    max_value=1.0,
    value=0.15,  # Default value
    format="%.2f"
)
#threshold for fc and foot off detection
# Force threshold input
Fthresh_fc_fo = st.sidebar.number_input(
    "Percentage of BW to detect foot contact/foot off (Fthresh_fc_fo)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,  # Default value
    format="%.2f"
)

save_file_path = st.sidebar.text_input("Enter the path for the folder to save file to", value='/Users/kayayoshida/Desktop/Dflow_proc_data/')


saveall = st.sidebar.checkbox("Save all files", value=False, key='saveall_checkbox')


# Multiple files uploader
filelist = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)
# filelist = sorted(glob.glob('/Users/kayayoshida/Documents/PhD/Study 1 - YA /DATA/raw dflow data/SP_31_10_2022_050008.txt'))
   
for file in filelist:   
        odf = pd.read_csv(file, header= 0, delimiter = "\t" )
        st.subheader(f"Uploaded file: {file.name}")
        # time = odf.iloc[:,0]
        
        # Extract time values after "#" rows
        time_series = odf.loc[(odf['Time'] != '#') & (odf['Time'].shift(-1) != '#'), 'Time']
        
        # Convert time_series to a Series
        time_series = pd.Series(time_series)
        time_series = time_series.astype(str)
        time_series = time_series[~time_series.str.contains('#')]
        time_series = time_series.astype(float)
        
        hash_indices = odf.index[odf['Time'] == '#']
        filename  = file.name
        if '000' in filename:
                filename = '_'.join(filename.split('000')[:-1])

        pid = filename[0:8]
        
        # Initialize an empty list to store time values
        time_values = []
        
        # Iterate through hash_indices and extract values immediately after them
        for idx in hash_indices:
            if idx + 1 < len(odf):  # Check if the index is within bounds
                time_value = odf.at[idx + 1, 'Time']
                if not pd.isna(time_value):  # Check if the value is not NaN
                    time_values.append(time_value)
        
        # Convert the list to a Series
        events = pd.Series(time_values)
        events = events[pd.to_numeric(events, errors='coerce').notnull()].reset_index(drop=True)
        
        
        # Convert the DataFrame to numeric, coercing non-numeric values to NaN
        odf_numeric = odf.apply(pd.to_numeric, errors='coerce')
       
        
        odf_numeric = odf_numeric.astype(float)
      
        # Drop rows containing NaN values   odf_dropped = odf_numeric.dropna().reset_index(drop=True)
    
        odf_dropped = odf_numeric.dropna().reset_index(drop=True)
       
        # odf = odf_dropped
        
        events = events.apply(pd.to_numeric, errors='coerce')
        
        matching_indices = []
        
        # Iterate over the events series
        for event_time in events:
            # Find indices where the Time column matches the event time
            indices = odf_dropped.index[odf_dropped['Time'] == event_time].tolist()
            # Append the indices to the matching_indices list
            matching_indices.extend(indices)
        
        
        # Create a Series of matching indices
        event_indices = pd.Series(matching_indices)
        
        events = pd.concat([event_indices, events], axis=1)
        df = odf_dropped.iloc[events.iloc[0,0]:,:].reset_index(drop=True) #crop data to first event - when walking starts
        df['Time'] = df['Time'] - df['Time'][0]
        zero_time = time_series.astype(float) - events.iloc[0,1]
        zero_start_time = zero_time.iloc[events.iloc[0,0]:].reset_index(drop=True)
        zero_time = zero_time.reset_index(drop=True)
        df = df.iloc[:len(zero_start_time),:]
        
        Fy = odf_dropped.iloc[:,[5,14]] 
        Fz = odf_dropped.iloc[:,[6,15]]

        Fy_zero = df.iloc[:,[5,14]] 
        
        #calculate bodyweight in N an Kg and force threshold for ouble support detection
        total_baseline_FY = Fy.iloc[:10000, 0] + Fy.iloc[:10000, 1] 
        Bw_N = (total_baseline_FY).mean()
        Bw_kg = Bw_N/9.81

        f"Body weight = {round(Bw_kg,2)} kg, {round(Bw_N,2)} N"
        bw_thresh = Bw_N*Fthresh
        bw_thresh_10 = Bw_N*.10
        
        f"bw threshold = {round(bw_thresh,2)} N, {Fthresh}% of BW"
        #calculate fast belt
        EDA_raw = odf_dropped.iloc[:,19]
        # EDA_raw.to_csv('/Users/kayayoshida/Desktop/test.csv')
        # EDA_raw = EDA_raw.dropna().reset_index(drop=True)
        lbs = odf_dropped.iloc[:,20]
        rbs = odf_dropped.iloc[:,22]
        if lbs.mean() > rbs.mean():
            print('Fast belt = left')
            fastbeltspeed = lbs
            slowbeltspeed = rbs
            leftfast = True
        elif lbs.mean() < rbs.mean():
            print('Fast belt = right')
            fastbeltspeed = rbs
            slowbeltspeed = lbs
            leftfast = False

        else:
            print('Tied condition')
            leftfast = 'Tied'
            
            
        #normalize Fy to boyweight
        Fy_norm = Fy/Bw_N
        Fz_norm = Fz/Bw_N
    
        
        #now calculate step length using starts of double supports as timing 
        
        LCOPz = df['FP1.CopX']*-1 #negate COPz signal as it is recorded backwards
        RCOPz = df['FP2.CopX']*-1
        
        #calculate step lengths between each event 
        events_zero = events.copy()
        events_zero.columns = ['Events (samp)', 'Events (sec)']
        events_zero = events_zero - events_zero.iloc[0,:]
        events_samp = events_zero['Events (samp)']
        #add end of session as an event:
        events_samp = pd.concat([events_samp, pd.Series([len(df)])]).reset_index(drop=True)
        #set the time col in df to start at zero
        

        SL_df = pd.DataFrame()
        dict_of_cons = {}
        dict_of_SL = {}
        dict_of_SW = {}
        SL_all = [] 
        SW_all = [] 
        SLS_all = []
    

        all_db_supp_start_times = pd.DataFrame()
        all_db_supp_end_times = pd.DataFrame()
        all_Fy_idx_end = pd.DataFrame()
        all_Fy_idx = pd.DataFrame()

        # Iterate through events_samp and create separate DataFrames in the dictionary for each condition
        for idx in range(len(events_samp) - 1):
            start_idx = events_samp[idx].astype(int)
            end_idx = events_samp[idx + 1].astype(int)
            dict_of_cons[f'DF_{idx+1}'] = df.iloc[start_idx:end_idx]
    
    #iterate through each condition based on event markers
        for key, con_df in dict_of_cons.items():
            
            # Extract relevant columns
            Fy_con = con_df.iloc[:, [0, 5, 14]]
            
            # Create boolean masks for rows where both Fy values are greater than the threshold
            mask = (Fy_con.iloc[:, 1] > bw_thresh) & (Fy_con.iloc[:, 2] > bw_thresh)
            
            # Use the mask to filter the DataFrame and get indices
            Fy_con_idx = Fy_con[mask]
            all_db_supp_idx = Fy_con.index[mask].tolist()
                    # Fy_con_idx #where 

    


            # testdf
            db_supp_idx = [] 
            db_supp_idx_end = [] 
            lfo_locs = []
            rfc_locs = []
            lfc_locs = []
            rfo_locs = []
            lfo_w_locs = []
            rfc_w_locs = []
            lfc_w_locs = []
            rfo_w_locs = []
            lsteptime = []
            rsteptime = []
            Lsly = []
            Rsly = []
            Lswy = []
            Rswy = []
            lfy_idx_start = []
            lfy_idx_end = []
            rfy_idx_start = []
            rfy_idx_end = []

            lfy_idx = []
            rfy_idx = []
            df_idx = pd.DataFrame()
            df_idx_end = pd.DataFrame()
          


            for j in range(len(all_db_supp_idx)): #get just the indices where the difference between consecutive samples is greater than 1 - new steps
                if all_db_supp_idx[j] - all_db_supp_idx[j-1] > 1:
                    db_supp_idx.append(all_db_supp_idx[j]) #starts of dbspt
                    db_supp_idx_end.append(all_db_supp_idx[j-1]) #ends of dbspt
            
        
                    df_idx = pd.concat([df_idx, Fy_con_idx.iloc[j,:].reset_index(drop=True)], axis=1).reset_index(drop=True) # get the starts of dbspt
                    df_idx_end = pd.concat([df_idx_end, Fy_con_idx.iloc[j-1,:].reset_index(drop=True)], axis=1).reset_index(drop=True) # get the starts of dbspt

            FY_idx = df_idx.T
            FY_idx = FY_idx.reset_index(drop=True)
            FY_idx.columns = ['Time', 'LFY', 'RFY']

            FY_idx_end = df_idx_end.T
            FY_idx_end = FY_idx_end.reset_index(drop=True)
            FY_idx_end.columns = ['Time', 'LFY', 'RFY']
            
            for k in range(len(db_supp_idx)): #get the COP locations at start and end of dbst
                    if df.iloc[db_supp_idx[k],5] > df.iloc[db_supp_idx[k],14]: #right step when LFY > RFY - right is initiating next step
                        lfo = df.iloc[db_supp_idx[k], 3] # get the COP location at that time
                        rfc = df.iloc[db_supp_idx[k], 12]
                        lfo_locs.append(lfo)
                        rfc_locs.append(rfc)
                        rtime = df.iloc[db_supp_idx[k],0]
                        lfo_w = df.iloc[db_supp_idx[k], 1] # get the COP location at that time
                        rfc_w = df.iloc[db_supp_idx[k], 10]
                        lfo_w_locs.append(lfo_w)
                        rfc_w_locs.append(rfc_w)
                        
                        rsl =  lfo - rfc #right step length = left foot off loc - right foot contact location
                        rsw = abs(lfo_w) + rfc_w #calculate step with
                        if rsl > stepthresh:
                            Rsly.append(rsl)
                            rsteptime.append(rtime) 
                            Rswy.append(rsw)
                    
                    if df.iloc[db_supp_idx[k],5] < df.iloc[db_supp_idx[k],14]: #left step
                        # print(Fy.iloc[k, 0], Fy.iloc[k, 1])
                        lfc = df.iloc[db_supp_idx[k], 3]
                        rfo = df.iloc[db_supp_idx[k], 12]
                        lfc_locs.append(lfc)
                        rfo_locs.append(rfo)
                        ltime = df.iloc[db_supp_idx[k],0]
                        lfc_w = df.iloc[db_supp_idx[k], 1] # get the COP location at that time
                        rfo_w = df.iloc[db_supp_idx[k], 10]
                        lfc_w_locs.append(lfc_w)
                        rfo_w_locs.append(rfo_w)
                        
                        
                        lsl =  rfo - lfc
                        lsw = rfo_w + abs(lfc_w)
                        if lsl > stepthresh:
                            Lsly.append(lsl)
                            lsteptime.append(ltime)
                            Lswy.append(lsw)
            
    
    
        
            db_supp_time = zero_start_time[db_supp_idx]
            all_db_supp_start_times = pd.concat([all_db_supp_start_times, db_supp_time], axis=0).reset_index(drop=True)
            
            db_supp_time_end = zero_start_time[db_supp_idx_end]
            all_db_supp_end_times = pd.concat([all_db_supp_end_times, db_supp_time_end], axis=0).reset_index(drop=True)
            
          
     

            con_SL_all= pd.concat([pd.Series(lsteptime), pd.Series(Lsly), pd.Series(rsteptime), pd.Series(Rsly)], axis=1, keys = ['LSLx', 'LSLy', 'RSLx', 'RSLy'])
            con_SW_all= pd.concat([pd.Series(lsteptime), pd.Series(Lswy), pd.Series(rsteptime), pd.Series(Rswy)], axis=1, keys = ['LSWx', 'LSWy', 'RSWx', 'RSWy'])
          
            allSLS = [] 
            #calculate SLS
            if len(Lsly) > len(Rsly):
                lensl = len(Rsly)
                SLSx = pd.Series(rsteptime)
            else:
                lensl = len(Lsly)
                SLSx = pd.Series(lsteptime)
                
            if leftfast == True:
                for b in range(lensl): #calculate symmetry
                  SLSy = (Lsly[b] - Rsly[b])/(Lsly[b] + Rsly[b])
                  allSLS.append(SLSy)
                  SLSser = pd.Series(allSLS)
                  fastbelt = 'Left belt fast'
            if leftfast == False:
                for b in range(lensl): #calculate symmetry
                  SLSy = (Rsly[b] - Lsly[b])/(Rsly[b] + Lsly[b])
                  allSLS.append(SLSy)
                  SLSser = pd.Series(allSLS)
                  fastbelt = 'Right belt fast'

            if leftfast == 'Tied':
                for b in range(lensl): #calculate symmetry
                  SLSy = (Rsly[b] - Lsly[b])/(Rsly[b] + Lsly[b])
                  allSLS.append(SLSy)
                  SLSser = pd.Series(allSLS)
                  fastbelt = 'Tied condition'
                
        
            SLSdf = pd.concat([SLSx, SLSser], axis=1, keys = ['SLSx', 'SLSy'])
            
            
            
            dict_of_SL[key] = con_SL_all
            dict_of_SW[key] = con_SW_all
            
            SL_all.append(con_SL_all)
            SW_all.append(con_SW_all)
            SLS_all.append(SLSdf)
             # Append con_SL_all DataFrame to SL_all list
            all_Fy_idx = pd.concat([all_Fy_idx, FY_idx], axis=0) 
            all_Fy_idx_end = pd.concat([all_Fy_idx_end, FY_idx_end], axis=0)     
        
        #calculate double support time using consecutive db supp idx
        # Concatenate all con_SL_all DataFrames into a single DataFrame SL_df
        SL_df = pd.concat(SL_all, ignore_index=True)
        SW_df = pd.concat(SW_all, ignore_index=True)
        RSL_df = SL_df.iloc[:,2:].reset_index(drop=True)
        LSL_df = SL_df.iloc[:,:2].reset_index(drop=True)
        SL_df = pd.concat([LSL_df, RSL_df], axis=1)
        RSW_df = SW_df.iloc[:,2:].reset_index(drop=True)
        LSW_df = SW_df.iloc[:,:2].reset_index(drop=True)
        SW_df = pd.concat([LSW_df, RSW_df], axis=1)
        
        SLS_df_all = pd.concat(SLS_all, ignore_index=True) # SLS for whole session
        
        all_db_supp_end_times = all_db_supp_end_times.iloc[1:].reset_index(drop=True)

        fig = go.Figure()
        
        # Add force data as a line plot
        fig.add_trace(go.Scatter(
            x=zero_start_time[:len(Fy)],
            y=Fy_zero.iloc[:,0],
            mode='lines',
            name='Lfy'
        ))
        
        fig.add_trace(go.Scatter(
            x=zero_start_time[:len(Fy)],
            y=Fy_zero.iloc[:,1],
            mode='lines',
            name='Rfy'
        ))
        fig.add_trace(go.Scatter(
            x=all_Fy_idx['Time'],
            y=all_Fy_idx['RFY'],
            mode='markers',
            name='Rfy db end'
        ))
         
        fig.add_trace(go.Scatter(
            x=all_Fy_idx['Time'],
            y=all_Fy_idx['LFY'],
            mode='markers',
            name='Lfy db start'
        ))
        
        fig.add_trace(go.Scatter(
            x=all_Fy_idx_end['Time'],
            y=all_Fy_idx_end['RFY'],
            mode='markers',
            name='Rfy db start'
        ))
         
        fig.add_trace(go.Scatter(
            x=all_Fy_idx_end['Time'],
            y=all_Fy_idx_end['LFY'],
            mode='markers',
            name='Lfy db end'
        ))
        for event in events_zero.iloc[:,1]:
            fig.add_vline(x=event, line_width=2, line_color="grey")

        for dbidx in all_db_supp_start_times:
            fig.add_vline(x=dbidx, line_width=1, line_color="pink")
        for idx in all_db_supp_end_times:
            fig.add_vline(x=idx, line_width=1, line_color="green")

        



        fig.update_layout(
            title=(f'Fy for both belts across full session with double support indices, used to calculate step length. {fastbelt} '),
            showlegend=True,
            template="simple_white",
            plot_bgcolor='white'
        )
        st.plotly_chart(fig)
            
    
        all_db_time_calc = pd.DataFrame()
        # #calculate db time from start and end
        for v in range(len(all_db_supp_end_times)):
            db_time_calc = all_db_supp_end_times.iloc[v] - all_db_supp_start_times.iloc[v]
            all_db_time_calc = pd.concat([all_db_time_calc, db_time_calc], axis=0)
            
       
        DBST_Time_Y = all_db_time_calc.reset_index(drop=True)
        DBST_Time_X =all_db_supp_start_times.reset_index(drop=True)
        len(DBST_Time_X)
        len(DBST_Time_Y)


        if len(DBST_Time_X) > len(DBST_Time_Y):
            DBST_Time_X = DBST_Time_X.iloc[:len(DBST_Time_Y)]


        if len(DBST_Time_X) < len(DBST_Time_Y):
            DBST_Time_Y = DBST_Time_Y.iloc[:len(DBST_Time_X)]

        dbst_df = pd.concat([DBST_Time_X.astype(float), DBST_Time_Y.astype(float)], axis=1, keys = ['DBST_Time_X', 'DBST_Time_Y'])
        dbst_df = dbst_df.iloc[1:,:]



        # Create subplots with shared x-axis
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=[f'Step lengths for {pid}', f'SLS for {pid}', f'EDA for {pid}'])
        
        # Add traces to the first subplot (Step lengths)
        fig.add_trace(go.Scatter(x=SL_df['LSLx'], y=SL_df['LSLy'], mode='markers', name='LSL'), row=1, col=1)
        fig.add_trace(go.Scatter(x=SL_df['RSLx'], y=SL_df['RSLy'], mode='markers', name='RSL'), row=1, col=1)
        # Add trace to the second subplot (EDA)
        fig.add_trace(go.Scatter(x=SLS_df_all['SLSx'], y=SLS_df_all['SLSy'], mode='lines', name='SLS'), row=2, col=1)
        
      
        # Add trace to the second subplot (EDA)
        fig.add_trace(go.Scatter(x=df['Time'], y=df['EDA'], mode='lines', name='EDA'), row=3, col=1)
        
          # Add vertical lines to both subplots
        for event in events_zero.iloc[:,1]:
            fig.add_vline(x=event, line_width=2, line_color="grey", row=1, col=1)
            fig.add_vline(x=event, line_width=2, line_color="grey", row=2, col=1)
            fig.add_vline(x=event, line_width=2, line_color="grey", row=3, col=1)
        
        # Update layout for the entire figure
        fig.update_layout(
            title=f'Step lengths and EDA for {pid}',
            showlegend=True,
            template="simple_white",
            plot_bgcolor='white'
        )
        
        # Update x-axis and y-axis titles for subplots
        fig.update_xaxes(title_text='Time (seconds)', row=2, col=1)
        fig.update_yaxes(title_text='Step length (m)', row=1, col=1)
        fig.update_yaxes(title_text='EDA (uS)', row=3, col=1)
        fig.update_yaxes(title_text='SLS', row=2, col=1)
        
        # Show the figure
        st.plotly_chart(fig)
        
        #need to calculate HS and TO - based on dflow logic
        #using FY data find the rising thresholds 
        def FC_FO_detect(thresh, side):
            
            # side = 'Right'
            if side == 'Left':
                col = 0
                
            if side == 'Right':
                col = 1
            force = Fy.iloc[events.iloc[0,0]:,col].reset_index(drop=True) # crop to first event
            force = force.dropna().reset_index(drop=True)
            force = force.astype(float)
            
            width = 1  # Example: Width of ±2 units around the threshold
    
            # Find indices where force crosses the threshold with the specified width for foot contact
            FC_indices = []
            for i in range(len(force)):
                if i > 0 and force[i] >= thresh and force[i-1] < thresh:
                    FC_indices.append(i - width) 
                    # print(force[i])
                # Start of the threshold crossing
                # elif force[i] < thresh and force[i-1] >= Fthresh:
                #     TO_indices.append(i + width)  # End of the threshold crossing
                    
                    
            FC_times = zero_start_time.iloc[FC_indices]
            
            
            #reverse signal to get foot off (FO)
            # force_rev = force[::-1].reset_index(drop=True)  
            
            FO_indices = []
            for i in range(len(force)):
                if i > 0 and force[i] <= thresh and force[i-1] > thresh:
                    FO_indices.append(i - width)  # Start of the threshold crossing
                # elif force_rev[i] < thresh and force_rev[i-1] >= Fthresh:
                #     TO_indices.append(i + width)  # End of the threshold crossing
                
            
            FO_times = zero_start_time.iloc[FO_indices]
            # FO_times = FO_times[::-1].reset_index(drop=True) 
    
            #graph forces with TO times
            fig = go.Figure()
            
            # Add force data as a line plot
            fig.add_trace(go.Scatter(
                x=zero_start_time[:len(force)],
                y=force,
                mode='lines',
                name='Force'
            ))
            
            # Add FC_times as scatter plot
            fig.add_trace(go.Scatter(
                x=zero_start_time.iloc[FC_indices],
                y=force.iloc[FC_indices],
                mode='markers',
                marker=dict(color='red'),
                name='Foot contact'
            ))
            
            
                    # Add FO_times as scatter plot
            fig.add_trace(go.Scatter(
                x=zero_start_time.iloc[FO_indices],
                y=force.iloc[FO_indices],
                mode='markers',
                marker=dict(color='green'),
                name='Foot off'
            ))
            
            # Set plot titles and labels
            fig.update_layout(
                title=f'FY with foot off and foot contact for {side} belt',
                xaxis_title='Time (Seconds)',
                yaxis_title='FY (N)',
                legend_title='Legend',
                    template="simple_white",
            plot_bgcolor='white'
        
            )
            
            st.plotly_chart(fig)

                
            
            FC_FO_times = pd.concat([FC_times.reset_index(drop=True), FO_times.reset_index(drop=True)],axis=1)
            FC_FO_times.columns = [f'{side} FC times', f'{side} FO times']
            FC_FO_indices = pd.concat([pd.Series(FC_indices), pd.Series(FO_indices)],axis=1)
            FC_FO_indices.columns = [f'{side} FC idx', f'{side} FO idx']
            return(FC_FO_times, FC_FO_indices)
                    
        
        

        bw_thresh_fc_fo = Bw_N*Fthresh_fc_fo
             
        
                        
        L_fc_fo_time, L_fc_fo_idx = FC_FO_detect(bw_thresh_fc_fo, 'Left')
        R_fc_fo_time, R_fc_fo_idx = FC_FO_detect(bw_thresh_fc_fo, 'Right')
        
        # dbst_df = dbst_df.iloc[1:,:]

        

        dbst_df = dbst_df.reset_index(drop=True)
        dbst_df.columns = range(dbst_df.shape[1])
        dbst_df.columns = ['DBST_Time_X', 'DBST_Time_Y']
         # graph double support time across session
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dbst_df['DBST_Time_X'], y=dbst_df['DBST_Time_Y'], mode='markers', name='DBST'))
        # Set plot titles and labels
        for event in events_zero.iloc[:,1]:
         fig.add_vline(x=event, line_width=2, line_color="grey")
        
        fig.update_layout(
            title=f'Double support time for {pid}',
            xaxis_title='Time (seconds)',
            yaxis_title='Double support time (seconds)',
            legend_title='Legend',    template="simple_white",
            plot_bgcolor='white'
        )
        
        fig.update_layout(
         yaxis=dict(
        range=[0, 1]
         )
        )

        st.plotly_chart(fig)



        #calculate step width



         # Create plot for step width
        fig = go.Figure()
        # Add traces to the first subplot (Step lengths)
        fig.add_trace(go.Scatter(x=SW_df['LSWx'], y=SW_df['LSWy'], mode='markers', name='LSW'))
        fig.add_trace(go.Scatter(x=SW_df['RSWx'], y=SW_df['RSWy'], mode='markers', name='RSW'))
        
         # Set plot titles and labels
        fig.update_layout(
            title=f'Step width',
            xaxis_title='Time (Seconds)',
            yaxis_title='Width (m)',
            legend_title='Legend',    template="simple_white",
            plot_bgcolor='white'
        )
        
        for event in events_zero.iloc[:,1]:
            fig.add_vline(x=event, line_width=2, line_color="grey")
    
        st.plotly_chart(fig)
    
        All_save_data = pd.concat([SL_df.reset_index(drop=True), SLS_df_all.reset_index(drop=True), zero_time.reset_index(drop=True), EDA_raw.reset_index(drop=True), zero_time.reset_index(drop=True), events_zero['Events (sec)'], L_fc_fo_time.reset_index(drop=True), R_fc_fo_time.reset_index(drop=True), lbs.reset_index(drop=True), rbs.reset_index(drop=True), SW_df['LSWy'].reset_index(drop=True), SW_df['RSWy'].reset_index(drop=True), dbst_df.reset_index(drop=True), Fz_norm.reset_index(drop=True), Fy_norm.reset_index(drop=True)], axis=1)
        All_save_data.columns = ['Left StepL X',	'Left StepL Y','Right StepL X',	'Right StepL Y',	'Symmetry Avg X',	'Symmetry Avg Y',	'EDA X',	'EDA Y',	'Time channel', 'Event indices', 	'L HS time', 	'L TO time',	'R HS time',	'R TO time',	'LBS',	'RBS',	'Left StepW Y',	'Right StepW Y',	'DBST Time X',	'DBST Time Y',	'L AP GRF norm',	'R AP GRF norm',	'L Fy norm',	'R Fy norm']
        # zero_time.to_csv(f'/Users/kayayoshida/Desktop/test2.csv')
        f"Data to save for {pid}"
        All_save_data 

        if '.txt' in filename:
                filename = '_'.join(filename.split('.txt')[:-1])

        # Unique keys for text inputs
        # path_key = f'save_file_path_{pid}'  # Unique key for file path input
        name_key = f'save_file_name_{pid}'  # Unique key for file name input

        # # Text inputs with unique keys
        # save_file_path = st.sidebar.text_input(
        #     "Enter the path for the folder to save file to",
        #     value='/Users/kayayoshida/Desktop/Dflow_proc_data/',
        #     key=path_key
        # )

        save_file_name = st.sidebar.text_input(
            "Enter the name for the processed file (without extension):",
            value=f"{filename}_Proc",
            key=name_key
        )
        if saveall:

            checkbox_key = f'save_checkbox_{pid}'  # Unique key for the checkbox
            widgetname = st.sidebar.checkbox("Save file", value=True, key=checkbox_key)



        else:
        # Checkbox to enable saving with a unique key
            checkbox_key = f'save_checkbox_{pid}'  # Unique key for the checkbox
            widgetname = st.sidebar.checkbox("Save file", value=False, key=checkbox_key)

        

        if widgetname:
            if save_file_path and save_file_name:
                # Save the DataFrame to CSV
                All_save_data.to_csv(f'{save_file_path}/{save_file_name}.csv', index=False)
                st.sidebar.write("File saved successfully.")
            else:
                st.sidebar.write("Please enter a valid path and file name.")
        else:
            st.sidebar.write("Check the box to enable file saving.")
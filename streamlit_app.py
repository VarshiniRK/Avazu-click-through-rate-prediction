
import streamlit as st
import pandas as pd
import pickle

#Loading up the Regression model we created

model = pickle.load(open('pipeline.pkl', 'rb'))

#Caching the model for faster loading
#@st.cache


# Define the prediction function
def predict(C1,banner_pos,site_id,site_domain,site_category, app_id, app_domain, app_category,device_id, device_ip, device_model, device_conn_type, C14, C15, C16, C18, C19, C20, C21, hour, day):
    def convert_obj_to_int(fm):
      object_list_columns = fm.columns
      object_list_dtypes = fm.dtypes
      print(object_list_columns)
      print(object_list_dtypes)
      for index in range(0,len(object_list_columns)):
        if object_list_dtypes[index] == object :
            fm[object_list_columns[index]] = fm[object_list_columns[index]].apply(lambda x: hash(x))
      return fm

    df=pd.DataFrame([[C1,banner_pos,site_id,site_domain,site_category, app_id, app_domain, app_category,device_id, device_ip, device_model, device_conn_type, C14, C15, C16, C18, C19, C20, C21, hour, day]], columns=['C1','banner_pos','site_id','site_domain','site_category', 'app_id', 'app_domain', 'app_category','device_id', 'device_ip', 'device_model', 'device_conn_type', 'C14', 'C15', 'C16', 'C18', 'C19','C20', 'C21', 'hour', 'day'])
    df_hashed = convert_obj_to_int(df)
    print(df_hashed.loc[0,:])
    print(df_hashed.dtypes)
    X = df_hashed.loc[:,:].to_numpy()
    print(X[0])
    prediction = model.predict(X)
    return prediction


st.title('Check if your ad will be clicked or not')
#st.image("""""")
st.header('Enter the characteristics of your ad:')
day = st.selectbox('Day of week:', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday'])
hour = st.number_input('Hour of day:', min_value=0, max_value=23, value=1)
C1 = st.selectbox('C1:', [1005, 1002, 1010, 1007, 1001, 1008, 1012])
banner_pos = st.selectbox('Banner_Position:', [0, 1, 5, 2, 4, 7, 3])

site_id = st.text_input('Site Id')
site_domain = st.text_input('Site Domain')
site_category = st.text_input('Site Category')
app_id = st.text_input('App Id')
app_domain = st.text_input('App Domain')
app_category = st.text_input('App Category')
device_id = st.text_input('Device Id')
device_ip = st.text_input('Device Ip')
device_model = st.text_input('Device Model')


device_conn_type = st.selectbox('Device Connection Type:', [0, 1, 5, 2, 4, 7, 3])
C15 = st.selectbox('C15:', [320,  300,  728,  216,  480, 1024,  768,  120])
C21 = st.number_input('C21',value=1)

C14 = st.number_input('C14',value=1)
C16 = st.number_input('C16',value=1)
C18 = st.number_input('C18',value=1)
C19 = st.number_input('C19',value=1)
C20 = st.number_input('C20',value=1)

if st.button('Check Click/Not click'):
    result = predict(C1,banner_pos,site_id,site_domain,site_category, app_id, app_domain, app_category,device_id, device_ip, device_model, device_conn_type, C14, C15, C16, C18, C19, C20, C21, hour, day)
    st.success('Prediction success', icon="✅")
    if result[0]==0:
      st.write('your ad will not be clicked')
    else:
      st.write('your ad will be clicked')
    

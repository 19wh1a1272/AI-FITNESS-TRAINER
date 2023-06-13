import streamlit as st
import cv2
from utils import *
import mediapipe as mp
import streamlit_authenticator as stauth
from types_of_exercise import TypeOfExercise
import yaml
import os
import pandas as pd
from datetime import datetime
from yaml.loader import SafeLoader

if not os.path.isdir('Tracking_data'):
    os.mkdir('Tracking_data')

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

if "webcam" not in st.session_state:    
    st.session_state.webcam = False

def use_webcam_callback():
    if st.session_state.webcam:
        st.session_state.webcam = False
    else:
        st.session_state.webcam = True

def main(username):
    if not os.path.isfile(os.path.join('Tracking_data',f'{username}.csv')):
        with open(os.path.join('Tracking_data',f'{username}.csv'),'w') as f:
            f.write('username,,,datetime,,,excersize\n')
    
    choice_type = st.selectbox(label='Status', 
                               options=['check status','Start exercise'], 
                               index=0)
    
    if choice_type == 'check status':
        if os.path.isfile('temp.txt'):
            with open('temp.txt','r') as f:
                data = f.readlines()
            data = data[0].strip()
            with open(os.path.join('Tracking_data',f'{username}.csv'),'a') as f:
                f.write(f'{data}\n')
        df = pd.read_csv(os.path.join('Tracking_data',f'{username}.csv'),sep=',,,')
        if not df.empty:
            st.dataframe(df[['datetime','excersize','counter']])
        else:
            st.text('No previous status')
    else:
        st.sidebar.subheader('Parameters')
        execrcise_type = st.sidebar.selectbox(label='Select exercise type', 
                                    options=['pull-up','sit-up','push-up','squat','walk'], 
                                    index=0)
        detection_confidence = st.sidebar.slider('Min Detection Confidence', 
                                                min_value =0.0,max_value = 1.0,
                                                value = 0.5)
        tracking_confidence = st.sidebar.slider('Min Tracking Confidence', 
                                                min_value =0.0,max_value = 1.0,
                                                value = 0.5)
            
        if (st.button('Use Webcam', on_click = use_webcam_callback())):
            if st.session_state.webcam:
                st.markdown(' ## Output')
                stframe = st.empty()
                vid = cv2.VideoCapture(0)
                vid.set(3, 800)  # width
                vid.set(4, 480)  # height
                with mp_pose.Pose(min_detection_confidence=detection_confidence,min_tracking_confidence=tracking_confidence) as pose:
                    counter = 0  # movement of exercise
                    status = True  # state of move
                    while vid.isOpened():
                        ret, frame = vid.read()
                        # result_screen = np.zeros((250, 400, 3), np.uint8)

                        frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
                        # recolor frame to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.flip(frame, 1)
                        frame.flags.writeable = False
                        # make detection
                        results = pose.process(frame)
                        # recolor back to BGR
                        frame.flags.writeable = True
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        try:
                            landmarks = results.pose_landmarks.landmark
                            counter, status = TypeOfExercise(landmarks).calculate_exercise(
                                execrcise_type, counter, status)
                        except:
                            pass

                        frame = score_table(execrcise_type, frame, counter, status)

                        # render detections (for landmarks)
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                thickness=2,
                                                circle_radius=2),
                            mp_drawing.DrawingSpec(color=(174, 139, 45),
                                                thickness=2,
                                                circle_radius=2),
                        )
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        stframe.image(frame,use_column_width=True)
                        with open('temp.txt','w') as f:
                            current_datetime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
                            f.write(f'{username},,,{current_datetime},,,{execrcise_type},,,{counter}\n')
                        
                    vid.release()
                    cv2.destroyAllWindows()
        
        else:
            st.text('Click on use_webcam to start capturing activity')

def login():
    c = st.sidebar.selectbox(label='Choose',options=['Login','Register'])
    if c == 'Login':
        name, authentication_status, username = authenticator.login('Login', 'main')
        if authentication_status:
            authenticator.logout('Logout', 'sidebar')
            st.markdown(f'Welcome **{name}**')
            st.title('Activity Detection App')
            st.sidebar.title('Activity Detection App')
            main(username)
        elif authentication_status is False:
            st.error('Username/password is incorrect')
        elif authentication_status is None:
            st.warning('Please enter your username and password')
    else:
        if st.session_state["authentication_status"]:
            st.markdown(f'You are already logged in as {st.session_state["name"]}')
        else:
            try:
                if authenticator.register_user('Register user', preauthorization=False):
                    st.success('User registered successfully')
                    with open('config.yaml', 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
            except Exception as e:
                st.error(e)
        
if __name__ == '__main__':
    login()
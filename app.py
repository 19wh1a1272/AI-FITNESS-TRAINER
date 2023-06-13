import streamlit as st
import cv2
from utils import *
import mediapipe as mp
import posemodule as pm
import time
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
    config['preauthorized'])

if "webcam" not in st.session_state:    
    st.session_state.webcam = False

def use_webcam_callback():
    if st.session_state.webcam:
        st.session_state.webcam = False
    else:
        st.session_state.webcam = True

def main(username):
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://img.freepik.com/free-photo/exercise-weights-iron-dumbbell-with-extra-plates_1423-223.jpg?w=1060&t=st=1684773885~exp=1684774485~hmac=90c2a8a5862a5b3ed25c1030bb5f302885cc1d65b47c05e692483a04175e802d ");
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    if not os.path.isfile(os.path.join('Tracking_data',f'{username}.csv')):
        with open(os.path.join('Tracking_data',f'{username}.csv'),'w') as f:
            f.write('username,,,duration,,,excersize,,,counter\n')
    
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
            st.table(df[['username','duration','excersize','counter']])
        else:
            st.text('No previous status')
    else:
        target_no = st.number_input(label='Enter target numbers',min_value=10)
        st.sidebar.subheader('Parameters')
        execrcise_type = st.sidebar.selectbox(label='Select exercise type', 
                                    options=['pull-up','sit-up','push-up','squat','walk','biceps'], 
                                    index=0)
        detection_confidence = st.sidebar.slider('Min Detection Confidence', 
                                                min_value =0.0,max_value = 1.0,
                                                value = 0.5)
        tracking_confidence = st.sidebar.slider('Min Tracking Confidence', 
                                                min_value =0.0,max_value = 1.0,
                                                value = 0.5)
        
        if (st.button('Use Webcam', on_click = use_webcam_callback())):
            if st.session_state.webcam:
                start_datetime = datetime.now()
                st.markdown(' ## Output')
                stframe = st.empty()
                vid = cv2.VideoCapture(0)
                detector = pm.poseDetector()
                vid.set(3, 800)  
                vid.set(4, 480)
                
                if execrcise_type == 'biceps':
                    count = 0
                    f=0
                    time.sleep(5)
                    while True :
                        ret, frame = vid.read()
                        img = detector.findPose(frame)
                        lmlist = detector.getPosition(img, draw= False)
                        
                        if len(lmlist)!=0:
                            cv2.circle(img,(lmlist[17][1],lmlist[17][2]),20,(0,0,255),cv2.FILLED)
                            cv2.circle(img,(lmlist[13][1],lmlist[13][2]),20,(0,255,0),cv2.FILLED) 
                            y1 = lmlist[17][2]
                            y2 = lmlist[13][2]
                            
                            length = y2-y1
                            if length>=0 and f==0:
                                f=1
                            elif length<0 and f==1:
                                f=0
                                count=count+1
                            if count >= target_no:
                                break
                            cv2.putText(img,"Count : "+str(int(count)),(70,50),cv2.FONT_HERSHEY_DUPLEX,1,(60,100,255),3)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            stframe.image(img,use_column_width=True)
                            with open('temp.txt','w') as f:
                                current_datetime = (datetime.now() - start_datetime)
                                f.write(f'{username},,,{current_datetime},,,{execrcise_type},,,{count+1}\n')
                    vid.release()
                    cv2.destroyAllWindows()
                else:
                    with mp_pose.Pose(min_detection_confidence=detection_confidence,min_tracking_confidence=tracking_confidence) as pose:
                        counter = 0  # movement of exercise
                        status = True  # state of move
                        while vid.isOpened():
                            ret, frame = vid.read()
                            # result_screen = np.zeros((250, 400, 3), np.uint8)

                            frame = cv2.resize(frame, (2000, 1000), interpolation=cv2.INTER_AREA)
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
                            if counter >= target_no:
                                break
                            with open('temp.txt','w') as f:
                                current_datetime = (datetime.now() - start_datetime)
                                f.write(f'{username},,,{current_datetime},,,{execrcise_type},,,{counter+1}\n')   
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
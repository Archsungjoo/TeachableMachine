import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Import the class labels from labels.txt and assign to a list
classes = [x.split(' ')[1].replace('\n', '') for x in open('labels.txt', 'r').readlines()]
# Load the Model
model = load_model('keras_model.h5', compile = False)

# Create the streamlit Title and camera_input
st.title(f'Architectural Heritage')
st.subheader(':red[근대건축유산] 의 보존과 활용을 위해')

st.divider()
st.image('building.png', caption='사라져가는 우리의 유산들')
st.subheader('왜 우리는 근대건축유산을 보존해야할까요?')
st.caption('첫째, 문화적 가치를 유지하기 위해')
st.caption('둘째, 역사적 중요성을 지니기 때문에')
st.caption('셋째, 지역사회의 정체성을 형성하기 위해')
st.caption('넷째, 지속가능한 발전과 활용을 위해')
st.caption('셋째, 미래세대를 위한 유산을 전달하기 위해')

st.divider()

st.subheader('보존에 함께해요!')
img_file_buffer = st.camera_input(f"평소 궁금하거나 분류해보고 싶은 건축물의 외부사진을 찍어주세요")


# Trigger when a photo has been taken and the bugger is no longer None
if img_file_buffer is not None:
    # Get the image and process it as required by the model
    # We are reshaping and converting the image to match the input the model requires.
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(cv2_img, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    probabilities = model.predict(image)

    # We now have the probabilities of the image being for either class
    # Check if either probability is over 80%, if so print the message for that classes.
    if probabilities[0,0] > 0.8:
        prob = round(probabilities[0,0] * 100,2)
        st.write(f"이 건물은 {prob}% 의 확률로 {classes[0]} 들이 지은 건축양식임이 확인되었습니다.")
    elif probabilities[0,1] > 0.8:
        prob = round(probabilities[0,1] * 100,2)
        st.write(f"이 건물은 {prob}% 의 확률로 {classes[1]} 들이 지은 건축양식임이 확인되었습니다.")
    else:
        st.write("죄송하지만, 건물의 양식을 파악할 수 없습니다. 다시 시도해주세요")

    # End on balloons
    st.balloons()

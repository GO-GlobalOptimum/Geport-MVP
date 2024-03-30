import streamlit as st
import os
import re
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from scipy.interpolate import interp1d
from sympy import symbols, evalf, sympify, latex
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

def create_Emotion_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
"""
    you are a helpful assistant that analysis emotion using our input and you must give us format is JSON,
    we determine JSON format which each of emotion is key that is string, percentage is value that is integer
    and you must Present an answer in a format that exists as JSON using json.loads()
"""
        ),
        HumanMessagePromptTemplate.from_template(
"""
### 예시 입력
Question: 이 블로그의 텍스트를 분석하여, 작성자가 글을 작성하며 경험했을 것으로 추정되는 주요 감정과 그 강도를 설명해주세요.
Context: (관련 내용이 있는 블로그 한개)

### 예시 출력
Answer : JSON 형식의 문자열

### 입력
Question: 이 블로그의 내용을 분석하여, 작성자가 경험했을 것으로 추정되는 감정과 그 감정의 강도를 설명해주세요. 
각 감정은 happy, joy, anxious, depressed, anger, sadness으로 구분해 설명하고, 각각의 감정이 글에서 어떻게 표현되었는지에 대한 예시를 포함해주세요.
Context: {context}

### 출력
Answer: """)])

def create_summary_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
you are a helpful summary assistant that make Great summary using only 1 sentence and short answer type
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
### 예시 입력
Question: 이 블로그 글을 요약해줄 수 있어?
Context: 오늘의 시드니 여행은 재미있었다. 오늘은 아침에 모닝빵과 스테이크를 먹었고 그 다음으로 근처 기념품관에서 오르골을 샀다.
    점심에는 바다가 보이는 레스토랑에서 와인과 함께 코스요리를 즐겼다. 저녁은 불꽃 축제와 함께 재미있는 친구들을 사귀며 놀았다. 
    이번 여행은 정말 재미있었고 다음에도 다시 왔으면 좋겠다.

### 예시 출력
오르골과 바다 그리고 불꽃축제와 함께 했던 정말 재미있었던 시드니 여행

### 입력
Question: 이 블로그 글을 요약해줘
Context: {context}

### 출력

            """
        )
])

# 워드 클라우드를 생성하는 함수
def create_wordcloud(data):
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=None)
    wordcloud.generate_from_frequencies(data)
    return wordcloud

def create_HappyKey_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
"""
 You are a useful helper that uses our input to analyze sentiment.
 You tell us what objects, food, etc. the user was happy with via the input. The format should be JSON that you must key is happy word that string, value is happy intensity that integer
 i will use your answer for JSON.format
"""
        ),
        HumanMessagePromptTemplate.from_template(
"""
### 예시 입력
Question: 이 블로그의 텍스트를 분석하여, 작성자가 어떠한 사물, 사람, 음식 등 행복감을 느꼈던 키워드가 무엇인지 단어를 추출해주세요(영어로 출력 바랍니다)
Context: (관련 내용이 있는 블로그 글들)

### 예시 출력
Answer : 블로그 내용들 중 행복감을 느끼게 했던 key word를 영어로 뽑아내서, 이의 강도를 같이 출력합니다.
Ex."dog": 80, "chicken": 70, "beef": 75, "taylor swift": 65, "day6": 60

### 입력
Question: 이 블로그의 내용을 분석하여, 작성자가 행복감을 느꼈던 요소 반드시 다섯개만 출력하도록.
작성자가 행복감을 느꼈던 요소를 분석 할 때, 문맥을 통해 분석해주길 바랍니다. "행복함","즐거움","기쁨" 이런식으로 행복과 연관된 직접적인 단어들이 아닌,
사물, 사람, 음식 등 "단어"를 추출해주세요. 이 단어들은 영어로 출력해주세요
그리고 요소로 인해 어느정도 행복했는지 각각의 강도를 백분율로 설명해주세요.
Context: {context}

### 출력
Answer:  """)])

def create_big5_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
"""
    You are a wise psychologist who must give us answers in json format (analyzing the Big 5 personality types),
    we determine JSON format which each of emotion is key that is string, score is value that is integer
    and you must Present an answer in a format that exists as JSON using json.loads()
"""
        ),
        HumanMessagePromptTemplate.from_template(
"""
    ### 예시 입력
    Question: 이건 사용자가 쓴 블로그 글이랑 사용자가 고른 big5 question의 문항 중 하나야. 이걸로 개방성, 성실성, 외향성, 우호성, 신경성에 대해 백분율 점수로 정리해줘
    answer: (user가 답한 대답들)
    Context: (user가 쓴 블로그 글)

    ### 예시 출력
    "openness" : 60,
    "sincerity" : 55,
    "extroversion" : 41,
    "friendliness" : 32,
    "neuroticism" : 60

    ### 입력
    Question: 이건 big5 question의 문항 중 하나야. 이걸로 개방성, 성실성, 외향성, 우호성, 신경성에 대해 백분율 점수로 정리해줘
    answer: {user}
    Context: {context}

    ### 출력
""")])

def create_final_summary_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
"""
    you are a helpful summary assistant that make Great summary, Through emotion, big5 information, happiness keywords, and blog posts,
    you help summarize and organize user information professionally. and you must speak korean. don't answer need more data
"""
        ),
        HumanMessagePromptTemplate.from_template(
"""
    ### 예시 입력
    Question: 다음과 같은 정보들로 사용자를 요약해줘
    emotion : ("happy" : 81, "sadness" : 21, "anger : 11, "joy" : 91, "depressed" : 1, "anxious" : 23)
    big : ("openness" : 60, "sincerity" : 55, "extroversion" : 41, "friendliness" : 76, "neuroticism" : 30 )
    word : ("dog": 80, "chicken": 70, "beef": 75, "taylor swift": 65, "day6": 60)
    Context: (user가 쓴 블로그 글)

    ### 예시 출력
    Answer: 종합적으로 다양한 감정 중 기쁨 감정이 상대적으로 높고 개나 치킨과 같은 단어가 행복함을 잘 표현하는 것으로 보아
    사용자는 개나 치킨 등의 귀엽고 맛있는 것을 좋아하는 것으로 보이며 블로그 글 등에서 확인 할 수 있었습니다.
    그리고 글의 작성 요령이나 big5 결과를 보았을 때 딱히 크게 예민하지 않고 외향적인 사람으로써 바캉스를 매우 잘 즐길 수 있는 성격이라고 
    보여집니다. 따라서 다양한 액티비티한 활동과 여행이 사용자의 행복감을 증가시켜줄 것이며 부정적인 감정들을 없애 줄 것입니다.

    ### 입력
    Question: 다음과 같은 정보들로 사용자를 요약해줘
    emotion : {emotion}
    big : {big5}
    word : {word}
    Context: {context}

    ### 출력
    Answer: 
""")])

def main():
    st.title('블로그로 시작하는 나만의 퍼스널 브랜딩, iGeport')  
    
    with st.form("geport_form"):
        st.title("나만의 iGeport 제작하기")
        st.text("Geport는 블로그 URL을 바탕으로 나만의 퍼스널 브랜딩을 제작합니다.\n다음 몇 가지 질문에 답변해주시면 Geport를 제작해드립니다.")
        name = st.text_input('이름')
        
        # 5개의 블로그 URL 입력
        st.title("일상 블로그 1일차")
        url1 = st.text_input('일상 블로그 1일차')

        st.title("일상 블로그 2일차")
        url2 = st.text_input('일상 블로그 2일차')

        st.title("일상 블로그 3일차")
        url3 = st.text_input('일상 블로그 3일차')

        st.title("일상 블로그 4일차")
        url4 = st.text_input('일상 블로그 4일차')

        questions = {
            "질문1": ["나는 상상력이 풍부하다", "나는 예술적 경험을 중요하게 생각한다.", "나는 아이디어를 떠올리는 일을 즐긴다."],
            "질문2": ["나는 목표를 달성하기 위해 노력하고, 포기하지 않는다.", "나는 업무나 과제를 미루는 경향이 있다."],
            "질문3": ["나는 사회적 활동에 참여하는 것을 피하곤 한다.", "나는 타인과의 상호작용에서 활력을 얻는다."],
            "질문4": ["나는 타인과의 관계에서 양보하는 편이다.", "나는 다른 사람과의 관계에 있어서 타인을 많이 생각하는 편이다.", "나는 다른 사람의 마음에 잘 공감하는 편이다."],
            "질문5": ["나는 종종 스트레스나 불안을 느낀다.", "나는 감정 상태가 자주 바뀌는 편이다."]
        }

        user_choices = {}

        for category, qs in questions.items():
            st.write(f"### {category}")
            option = st.radio(f"{category}에 대한 질문:", options=qs, key=category)
            user_choices[category] = option
            
        # 제출 버튼
        submitted = st.form_submit_button("iGeport 제작하기")

        if submitted:
            # 1. vector store 생성
            # 프롬프트 생성
            url_list = [url1, url2,url3, url4]
            loader = WebBaseLoader(url_list)

            blog_content = loader.load()
            blog_content_question = "\n".join(user_choices.values())

            loader_1day = WebBaseLoader(url1)
            blog_content1 = loader_1day.load()

            loader_2day = WebBaseLoader(url2)
            blog_content2 = loader_2day.load()

            loader_3day = WebBaseLoader(url3)
            blog_content3 = loader_3day.load()

            loader_4day = WebBaseLoader(url4)
            blog_content4 = loader_4day.load()

            prompt = create_HappyKey_prompt().format_prompt(context=blog_content).to_messages()
            prompt_question = create_big5_prompt().format_prompt(context=blog_content, user=blog_content_question).to_messages()

            prompt_1day = create_Emotion_prompt().format_prompt(context=blog_content1).to_messages()
            prompt_2day = create_Emotion_prompt().format_prompt(context=blog_content2).to_messages()
            prompt_3day = create_Emotion_prompt().format_prompt(context=blog_content3).to_messages()
            prompt_4day = create_Emotion_prompt().format_prompt(context=blog_content4).to_messages()

            prompt_1day_summary = create_summary_prompt().format_prompt(context=blog_content1).to_messages()
            prompt_2day_summary = create_summary_prompt().format_prompt(context=blog_content2).to_messages()
            prompt_3day_summary = create_summary_prompt().format_prompt(context=blog_content3).to_messages()
            prompt_4day_summary = create_summary_prompt().format_prompt(context=blog_content4).to_messages()

            # llm35 인스턴스를 사용하여 감정 분석 요청

            responseHappyKey = llm35(prompt)
            emotion_response_1day = llm35(prompt_1day)
            emotion_response_2day = llm35(prompt_2day)
            emotion_response_3day = llm35(prompt_3day)
            emotion_response_4day = llm35(prompt_4day)

            summary_response_1day = llm35(prompt_1day_summary)
            summary_response_2day = llm35(prompt_2day_summary)
            summary_response_3day = llm35(prompt_3day_summary)
            summary_response_4day = llm35(prompt_4day_summary)

            emotion_data_1day = json.loads(emotion_response_1day.content)
            emotion_data_2day = json.loads(emotion_response_2day.content)
            emotion_data_3day = json.loads(emotion_response_3day.content)
            emotion_data_4day = json.loads(emotion_response_4day.content)

            response_question_data = llm35(prompt_question)

            response_question_data = json.loads(response_question_data.content)
            data_question = pd.DataFrame(list(response_question_data.items()), columns=['Personality Trait', 'Score'])

            prompt_final = create_final_summary_prompt().format_prompt(
                emotion = prompt_1day + prompt_2day + prompt_3day + prompt_4day,
                big5 = response_question_data,
                word = responseHappyKey,
                context = prompt_1day_summary + prompt_2day_summary + prompt_3day_summary + prompt_4day_summary
                ).to_messages()

            final_response = llm35(prompt_final)
                
            
            st.markdown("### 1일차 요약")
            st.write(summary_response_1day.content)

            st.markdown("### 2일차 요약")
            st.write(summary_response_2day.content)

            st.markdown("### 3일차 요약")
            st.write(summary_response_3day.content)

            st.markdown("### 4일차 요약")
            st.write(summary_response_4day.content)

            fig1, ax1 = plt.subplots()
            emotions = {"happy": [], "joy": [], "anxious": [], "depressed": [], "anger": [], "sadness": []}
            emotion_SOS = {"anxious": [], "depressed": [], "anger": [], "sadness": []}

            for day in [emotion_data_1day, emotion_data_2day, emotion_data_3day, emotion_data_4day]:
                for emotion, value in day.items():
                    try:
                        emotions[emotion].append(value)
                        if emotion == "anxious" or emotion == "anger" or emotion == "sadness" or emotion == "depressed" :
                            emotion_SOS[emotion].append(value)
                    except:
                        continue
            
            x = np.arange(len(emotions['happy']))  # 원래의 x 축 포인트
            x_new = np.linspace(0, len(emotions['happy']) - 1, 300)

            for emotion, values in emotions.items():
                if emotion == "happy" or emotion == "anger" or emotion == "sadness" or emotion == "joy" :
                    f = interp1d(x, values, kind='cubic')  # 'cubic' 보간 사용
                    ax1.plot(x_new, f(x_new), label=emotion, linestyle='-')

            # 그래프 설정
            ax1.set_xlabel('day')  # x축 레이블
            ax1.set_ylabel('emotion %')  # y축 레이블
            ax1.set_title('emotion each day')  # 그래프 제목
            ax1.legend()  # 범례 표시
            
            # x축 눈금 레이블 설정
            ax1.set_xticks(range(4))
            ax1.set_xticklabels(['1day', '2day', '3day', '4day'])

            # 그래프 표시
            st.markdown("### 감정 곡선")
            st.pyplot(fig1)

            emotions_avg = {emotion: np.mean(values) for emotion, values in emotion_SOS.items()}

            # 그래프를 그리기 위한 데이터 준비
            labels = list(emotions_avg.keys())
            means = list(emotions_avg.values())

            # 가로형 막대 그래프 생성
            fig2, ax2 = plt.subplots()
            ax2.barh(labels, means, color=['blue', 'orange', 'green', 'red'])

            ax2.set_xlabel('Average Score')
            ax2.set_xticks(np.arange(0, 101, 10))
            ax2.set_title('Average Emotional Scores')

            st.markdown("### 감정 SOS")
            st.pyplot(fig2)
            
            st.title(f'{name}님의 I-Geport')
            st.text("당신은 이것들 때문에 행복했었어요!")
            # 워드 클라우드 생성
            # st.text(responseHappyKey.content)
            data = json.loads(responseHappyKey.content)
            wordcloud = create_wordcloud(data)
            fig3, ax3 = plt.subplots()
            ax3.imshow(wordcloud, interpolation='bilinear')
            ax3.axis('off')

            # Streamlit에 차트를 보여줌
            st.pyplot(fig3)

            st.write("## big5 점수")
            # Plotting
            fig4, ax4 = plt.subplots()
            ax4.bar(data_question['Personality Trait'], data_question['Score'], color=['skyblue', 'orange', 'green', 'red', 'purple'])

            # Here you can adjust the fontsize of the x-axis labels
            ax4.set_xticklabels(data_question['Personality Trait'], rotation=45, fontsize=10)

            # Adding labels and title
            plt.xlabel('Personality Traits', fontsize=12)
            plt.ylabel('Scores', fontsize=12)
            plt.title('Personality Profile', fontsize=14)

            # Display the plot in Streamlit
            st.pyplot(fig4)

            st.write("## iGeport 최종 분석")
            st.write(final_response.content)

if __name__ == '__main__':
    main()
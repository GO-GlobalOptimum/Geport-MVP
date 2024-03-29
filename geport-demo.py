import streamlit as st
import os
import re
import matplotlib.pyplot as plt
import json
import numpy as np
from sympy import symbols, evalf, sympify, latex
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

def create_rag_prompt(type):
    if type == 1:
        return ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(
"""
You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,500 characters based on the User Answer and Context. We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective.

### 예시 입력
Question: 당신은 어떤 사람이 되고싶나요?
User Answer: 나만의 기록 친구, Looi라는 앱을 개발하면서, 많은 사람들이 기록을 통해서 심리적 불안감을 해소하고 있다는 것을 느꼈습니다. 이러한 경험을 통해서 더 많은 사람들이 자신의 감정을 기록하고, 공유하며 내면이 건강해지는 세상을 만들고 싶습니다.
Context: (관련 내용들이 담긴 문서)

### 예시 출력
Answer: 저의 스타트업 프로젝트인 Looi 앱을 통해, 많은 분들이 자신의 일상과 감정을 기록함으로써 심리적 안정감을 찾고 계시다는 것을 직접 목격하였습니다. 이러한 경험은 저에게 매우 큰 영감을 주었으며, 이를 통해 더 많은 분들이 자신의 감정을 솔직하게 기록하고, 이를 통해 자기 자신과 타인과의 깊은 연결을 경험할 수 있는 세상을 만드는 것이 저의 큰 소망이 되었습니다. 저는 기술이 단순히 생활을 편리하게 하는 도구를 넘어, 우리 각자의 내면을 들여다보고 성찰할 수 있는 강력한 수단이 될 수 있다고 믿습니다. 이러한 비전을 바탕으로, 저는 앞으로도 사용자들이 자신의 내면을 건강하게 다스리고, 서로의 경험을 공유하며 서로를 이해하는 데 도움을 줄 수 있는 다양한 프로젝트를 개발해 나갈 계획입니다. 이 과정에서 저는 기술과 인간의 감정이 서로 조화롭게 어우러질 수 있는 방법을 모색하며, 사람들이 자신의 감정을 건강하게 표현하고 관리할 수 있는 더 많은 기회를 제공하기 위해 노력할 것입니다.

### 입력
Question: 당신은 어떤 사람이 되고싶나요?
User Answer: {answer}
Context: {context}

### 출력
Answer: """)])
    
    elif type == 2:
        return ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(
"""
You are an assistant who will help you with personal branding based on your blog. An answer to the question is created in Korean with approximately 1,500 characters based on the User Answer and Context. We plan to use the generated sentences as content to express ourselves, so please write from the user's perspective. Don't say thank you at the end of a sentence. Please write from the user's perspective.

### 예시 입력
Question: 당신은 좌우명은 무엇인가요?
User Answer: "세상에 안 되는 일은 없다. 안 된다고 생각해서 시도하지 않았기 때문에 안 되는 것일 뿐이다." 라는 좌우명을 가지고 있습니다. 개발을 잘 하지 못했지만 Looi를 개발하면서, 불가능해 보이는 많은 일들을 마주했습니다. 하지만 사용자에게 꼭 필요한 기능이라고 생각하면서 끊임없이 도전했고 그 결과, 성공적으로 서비스를 제작했으며 많은 사용자들이 Looi를 통해 자신의 감정을 기록하고, 공유하며 서로를 이해하는 데 도움을 받고 있습니다.
Context: (관련 내용들이 담긴 문서)

### 예시 출력
Answer: 세상에 안 되는 일은 없다"는 좌우명을 마음속 깊이 새기고, 저는 Looi 앱 개발이라는 새로운 여정을 시작하였습니다. 초기 개발 과정에서 많은 어려움에 부딪혔음에도 불구하고, 저는 사용자분들에게 꼭 필요한 기능을 제공하고자 하는 일념 하에 끊임없이 노력하였습니다. 그 결과, Looi는 성공적으로 출시되었으며, 많은 분들이 이를 통해 자신의 감정을 기록하고 공유함으로써 서로를 더 잘 이해하게 되었다는 소식을 접하게 되었습니다. 이는 저에게 큰 만족감과 보람을 안겨주었습니다.\n\n이러한 경험을 통해, 저는 도전을 두려워하지 않는 마음가짐의 중요성을 깨달았습니다. 앞으로도 저는 계속해서 새로운 도전에 맞서며, 이를 통해 배우고 성장해 나가고자 합니다. 또한, 저는 더 많은 분들이 자신의 내면을 탐색하고 서로를 이해하는 데 도움이 될 수 있는 방법을 모색할 계획입니다.

### 입력
Question: 당신은 좌우명은 무엇인가요?
User Answer: {answer}
Context: {context}

### 출력
Answer: """)])
    

graph_prompt = """
Based on the user's information, we create a formula that can be drawn on a coordinate plane where the x-axis represents time and the y-axis represents the success index.
Please include anything that could signify an inflection point in your life. Basically, the formula should be structured in such a way that it increases over time. Please make it a function of degree 3 or higher and include one symbols such as sin, cos, tan, and ln.
The response is returned in json format, with the first key being "equation" and the second key being "explanation". explanation은 한국어로 대답해주세요. 사용자의 입장에서 수식에 대한 의미를 설명하는 것입니다. The more complex the formula, the better. but, it must not contain complex numbers. create only formula ex) y = 1.5 * ln(x + 1) + 0.5 * e**(0.1 * (x - 20))
"""


def url_to_text(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    return splits

def create_vector_store(url_list):
    flag = False
    for url in url_list:
        if not flag:
            docs = url_to_text(url)
            flag = True
        else:
            docs += url_to_text(url)

    splits = split_text(docs)

    vector_store = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    return retriever

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, prompt):
    rag_chain = (
        {"context": retriever | format_docs, "answer": RunnablePassthrough()}
        | prompt
        | llm35
        | StrOutputParser()
    )
    return rag_chain


def main():
    st.title('블로그로 시작하는 나만의 퍼스널 브랜딩, Geport')  
    
    with st.form("geport_form"):
        st.title("나만의 Geport 제작하기")
        st.text("Geport는 블로그 URL을 바탕으로 나만의 퍼스널 브랜딩을 제작합니다.\n다음 몇 가지 질문에 답변해주시면 Geport를 제작해드립니다.")
        name = st.text_input('이름')
        
        # 5개의 블로그 URL 입력
        st.title("블로그 URL 입력하기")
        st.text("블로그 URL을 입력하시면, 해당 블로그 내용을 분석하여 Geport 제작에 활용합니다.")
        url1 = st.text_input('블로그 URL 1')
        url2 = st.text_input('블로그 URL 2')
        url3 = st.text_input('블로그 URL 3')
        url4 = st.text_input('블로그 URL 4')
        url5 = st.text_input('블로그 URL 5')

        # 질문 입력
        st.title('퍼스널 브랜딩을 위한 질문')
        st.text("첨부된 블로그의 내용을 바탕으로 간단하게 다음 질문에 대해 작성해주시면 Geport를 제작해드립니다.\n단, 반드시 블로그에 작성된 내용과 일치할 필요는 없습니다. 자유롭게 작성해주세요.")
        
        st.markdown("### 당신은 어떤 사람이 되고싶나요?")
        question_1 = st.text_area('예) 나만의 기록 친구, Looi라는 앱을 개발하면서, 많은 사람들이 기록을 통해서 심리적 불안감을 해소하고 있다는 것을 느꼈습니다. 이러한 경험을 통해서 더 많은 사람들이 자신의 감정을 기록하고, 공유하며 내면이 건강해지는 세상을 만들고 싶습니다.')
        
        st.markdown("### 당신의 좌우명은 무엇인가요?")
        question_2 = st.text_area('예) "세상에 안 되는 일은 없다. 안 된다고 생각해서 시도하지 않았기 때문에 안 되는 것일 뿐이다." 라는 좌우명을 가지고 있습니다. 개발을 잘 하지 못했지만 Looi를 개발하면서, 불가능해 보이는 많은 일들을 마주했습니다. 하지만 사용자에게 꼭 필요한 기능이라고 생각하면서 끊임없이 도전했고 그 결과, 성공적으로 서비스를 제작했으며 많은 사용자들이 Looi를 통해 자신의 감정을 기록하고, 공유하며 서로를 이해하는 데 도움을 받고 있습니다.')

        st.markdown("### 당신이 좋아하는 것은 무엇인가요?")
        question_3 = st.text_area('예) 문제를 정확하게 분석하여 해결하는 것을 좋아합니다. 그래서 문제의 근본 원인을 파악하여 문제를 정의하기 위해 노력합니다.')

        st.markdown("### 당신이 잘 하는 것은 무엇인가요?")
        question_4 = st.text_area('예) 소셜리스닝을 잘 합니다. 다양한 분야에 관심이 많아서 많은 사람의 의견에 귀를 기울이고, 저만의 의견을 정립하려고 노력합니다.')

        st.markdown("### 인생의 변곡점은 무엇인가요? 힘들었지만 극복했던 경험을 알려주세요.")
        question_5 = st.text_area('예) 대학교 2학년 때, 학교 생활에 흥미를 느끼지 못하고 많이 방황했던 경험이 있습니다. 하지만 서비스를 제작하는 과제를 진행하면서 몇날 며칠 밤을 새워가며 프로젝트를 완성했고, 그 결과로 성공적으로 프로젝트를 마무리했습니다. 이 과정이 힘들지 않았고, 오히려 학교생활에 대한 흥미를 되찾게 해주었습니다.')

        # 제출 버튼
        submitted = st.form_submit_button("Geport 제작하기")

        if submitted:
            # 1. vector store 생성
            url_list = [url1, url2, url3, url4, url5]
            retriever = create_vector_store(url_list)

            # 2. RAG chain 생성 (1), (2)
            rag_chain_1 = create_rag_chain(retriever, create_rag_prompt(1))
            rag_chain_2 = create_rag_chain(retriever, create_rag_prompt(2))

            # 3. RAG chain 실행 Q1 -> (1), Q2 -> (2)
            answer_1 = rag_chain_1.invoke(question_1)
            answer_2 = rag_chain_2.invoke(question_2)

            # 4. Q3, Q4, (2)를 활용하여 (2)를 다시 생성
            answer_2 = llm35.predict(f"Don't say thank you at the end of a sentence. Please write from the user's perspective. 좌우명을 설명하는 글을 존댓말로 작성해주세요. 저의 인생의 좌우명은 ~ 입니다. 로 문장을 시작해주고, 사용자의 정보를 바탕으로 좌우명을 설명하는 글을 작성해주세요.(1000자)\n\n사용자의 좋아하는 것은 {question_3}\n사용자가 잘하는 것은 {question_4}\n\n{answer_2}.")

            # 5. Q5를 활용하여 (3)을 생성
            answer_3 = llm35.predict(f"Don't say thank you at the end of a sentence. Please write from the user's perspective. 인생의 변곡점을 설명하는 글을 존댓말로 작성해주세요.(1000자) 사용자의 정보는 다음과 같습니다. 다음 내용을 참고하여 새로 작성해주세요.\n\n{answer_2}\n{question_5}")

            # 6. (1), (3)을 활용하여 (4)를 생성
            answer_4 = llm35.predict(f"{graph_prompt}\n{answer_1}\n{answer_3}")

            # 7. (2), (4)를 활용하여 (5)를 생성
            answer_5 = llm35.predict(f"Don't say thank you at the end of a sentence. Please write from the user's perspective. 한줄 요약과 발전하기 위한 각오, 포부 등을 담은 글을 존댓말로 작성해주세요.(1000자)\n\n사용자의 정보는 다음과 같습니다.\n\n{answer_1}\n\n{answer_2}\n\n{json.loads(answer_4)['explanation']}")

            # 8. 결과 출력
            st.title(f'{name}님의 Geport')

            st.markdown("### 저는 이런 사람이 되고싶어요")
            st.write(answer_1) # TODO: 제목 생성?

            st.markdown("### 저의 좌우명은 이렇습니다")
            st.write(answer_2)

            st.markdown("### 제 인생의 변곡점은 이겁니다")
            st.write(answer_3)

            st.markdown("### 저의 인생을 수식으로 표현하면 이렇습니다")

            # sympy를 사용하여 수식 파싱
            x = symbols('x')
            formula = json.loads(answer_4)["equation"]
            equation = formula.split('=')[1].strip()
            if 'if' in equation:
                equation = equation.split('if')[0].strip()
            equation = equation.replace('^', '**')
            equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
            equation = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', equation)

            # 수식 파싱
            formula = sympify(equation)
            latex_formula = latex(formula)

            # x 값 정의 (예: 0부터 50까지)
            x_vals = np.linspace(0, 50, 400)

            # 수식 평가 및 y 값 계산
            y_vals = [formula.subs(x, val).evalf() for val in x_vals]

            # 그래프 그리기
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals)

            # 그래프 스타일 설정
            ax.set_xlabel("x time")
            ax.set_ylabel("y success index")
            ax.set_title(f"Life Formula (${latex_formula}$)")

            ax.set_xticks([])  # x축 눈금 제거
            ax.set_yticks([])  # y축 눈금 제거

            # Streamlit에 그래프 출력
            st.pyplot(fig)

            st.write(json.loads(answer_4)["explanation"])

            st.markdown("### Geport 솔루션")
            st.write(answer_5)


if __name__ == '__main__':
    main()
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.status import HTTP_422_UNPROCESSABLE_ENTITY,HTTP_200_OK
from langchain.llms.openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader

from utility.privateDocSummaryUtility import *
from PyPDF2 import PdfReader

# Create your views here.
class Summarizedoc(APIView):
    def post(self, request):
        try:
            print("in api")
            print(request)
            data = request.data
            
            model = data.get('model')
            print(model)
            api_key = data.get('api_key')
            num_words = data.get('num_words')
            doc_length = data.get('doc_length')
            file = request.FILES['file']
            print(request.data)
            file_type = file.name.split('.')[1]
            print(file_type)
            if file_type not in ['txt','pdf']:
                return Response({'status_code':400,'message':'Please upload text or pdf file','data':''},status=HTTP_422_UNPROCESSABLE_ENTITY)    

            if model == 'ChatGPT':
                llm = OpenAI(temperature=0,openai_api_key=api_key)
            elif model == 'Gemini-Pro':
                llm = ChatGoogleGenerativeAI(google_api_key=api_key,model='gemini-pro',temperature=0)

            if file_type == 'txt':
                file_data = file.read().decode('utf-8')
                print(file_data)
            else:
                reader = PdfReader(file) 
                print(reader)
                file_data = []
                print(file_data)
                print(len(reader.pages))
                for i in range(len(reader.pages)):
                    page = reader.pages[i]
                    text = page.extract_text()
                    file_data.append(text)
                file_data = " ".join(file_data)

            print(file_data)

            num_tokens = llm.get_num_tokens(file_data)
            if num_tokens > 20000 and doc_length.lower() in ['short','medium']:
                return Response({'status_code':400,'message':'Document is long, Please select document length Large','data':''},status=HTTP_422_UNPROCESSABLE_ENTITY)     

            print(llm.get_num_tokens(file_data))
            print("llm",llm)
            if doc_length.lower() in ['short','medium']:
                summary = summarize_doc(model,llm,file_data)
            else:
                print("Large")
                summary = summarize_long_doc(model,llm,api_key,file_data)

            if summary != 1:
                return Response({'status_code':200,'message':'success','data':summary},status=HTTP_200_OK)
            else:
                return Response({'status_code':400,'message':'Document is not long enough, Please select document length short or Medium','data':''},status=HTTP_422_UNPROCESSABLE_ENTITY)
        except Exception as e:
            return Response({'status_code':400,'message':e,'data':''},status=HTTP_422_UNPROCESSABLE_ENTITY)
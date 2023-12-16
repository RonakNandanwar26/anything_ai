from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.status import HTTP_422_UNPROCESSABLE_ENTITY,HTTP_200_OK
from langchain.llms.openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader

from utility.privateDocUtility import *
from PyPDF2 import PdfReader
import logging

logger = logging.getLogger(__name__)


# Create your views here.
class Summarizedoc(APIView):
    def post(self, request):
        logging.info("in api")
        logging.info(request)
        data = request.data
        
        model = data.get('model')
        logging.info(model)
        api_key = data.get('api_key')
        # num_words = data.get('num_words')
        doc_length = data.get('doc_length')
        file = request.FILES['file']
        logging.info(request.data)
        file_type = file.name.split('.')[1]
        logging.info(file_type)
        if file_type not in ['txt','pdf']:
            return Response({'status_code':400,'message':'Please upload text or pdf file','data':''},status=HTTP_422_UNPROCESSABLE_ENTITY)    

        if model == 'ChatGPT':
            llm = OpenAI(temperature=0,openai_api_key=api_key)
        elif model == 'Gemini-Pro':
            llm = ChatGoogleGenerativeAI(google_api_key=api_key,model='gemini-pro',temperature=0)
        
        file_data = get_file_data(file,file_type)
        # if file_type == 'txt':
        #     file_data = file.read().decode('utf-8')
        #     print(file_data)
        # else:
        #     reader = PdfReader(file) 
        #     print(reader)
        #     file_data = []
        #     print(file_data)
        #     print(len(reader.pages))
        #     for i in range(len(reader.pages)):
        #         page = reader.pages[i]
        #         text = page.extract_text()
        #         file_data.append(text)
        #     file_data = " ".join(file_data)

        num_tokens = llm.get_num_tokens(file_data)
        logging.info(num_tokens)
        if num_tokens > 20000 and doc_length.lower() in ['short','medium']:
            return Response({'status_code':400,'message':'Document is long, Please select document length Large','data':''},status=HTTP_422_UNPROCESSABLE_ENTITY)     

        logging.info("llm",llm)
        if doc_length.lower() in ['short','medium']:
            summary = summarize_doc(model,llm,file_data)
        else:
            logging.info("Large")
            summary = summarize_long_doc(model,llm,api_key,file_data)

        if summary != 1:
            return Response({'status_code':200,'message':'success','data':summary},status=HTTP_200_OK)
        else:
            return Response({'status_code':400,'message':'Document is not long enough, Please select document length short or Medium','data':''},status=HTTP_422_UNPROCESSABLE_ENTITY)





class QnADoc(APIView):
    def post(self,request):
        data = request.data
        model = data.get('model')
        api_key = data.get('api_key')
        question = data.get('question')
        file = request.FILES['file']
        file_name = file.name
        file_type = file.name.split('.')[1]
        file_data = get_file_data(file,file_type)

        response = doc_qna(model,api_key,question,file_name,file_data)
        return Response({'status_code':200,'message':'success','data':response},status=HTTP_200_OK)
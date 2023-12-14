import requests

def api_call(api_url,request_type,data=None,files=None,headers=None):
    
    if request_type == 'post':
        response = requests.post(api_url,data=data,files=files,headers=headers)

    
    return response
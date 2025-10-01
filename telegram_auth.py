import requests
import os
import json
from dotenv import load_dotenv

last_request_response = None

"Авторизация через telegram API"

load_dotenv()

HEADERS = {
    'Authorization': f'Bearer {os.getenv("TOKEN")}',
    'Content-Type': 'application/json'
}

# Function to query the API
def post_request_status(endpoint, json_body):
    url = f'{os.getenv("BASE_URL")}{endpoint}'
    response = requests.post(url, headers=HEADERS, json=json_body)
    if response.status_code == 200:
        response_json = response.json()
        if response_json.get('ok'):
            res = response_json.get('result', {})
            return res
        else:
            error_message = response_json.get('error', 'Unknown error')
            print(f"Error: {error_message}")
            return None
    else:
        print(f"Failed to get request status: HTTP {response.status_code}")
        return None

def send_message(phone_number):
    endpoint = 'sendVerificationMessage'
    json_body = {
        'phone_number': phone_number,         # Must be the one tied to request_id
        'code_length': 6,              # Ignored if you specify your own 'code'
        'ttl': 60,                     # 1 minute
        'payload': 'my_payload_here',  # Not shown to users
        'callback_url': 'https://my.webhook.here/auth'
    }
    response = post_request_status(endpoint, json_body)
    print(response)
    return response

def check_message(response, code):
    endpoint = 'checkVerificationStatus'
    json_body = {
        'request_id': response.get('request_id'), # Relevant request id
        'code': code,            # The code the user entered in your app
    }
    result = post_request_status(endpoint, json_body)
    status = result.get('verification_status', {}).get('status')
    return status == 'code_valid'


def verify_code(response, verification_code):
    return check_message(response, verification_code)
        


#res = send_message('79166844293')
#print(check_message((res['request_id'], input("Input code: "))))

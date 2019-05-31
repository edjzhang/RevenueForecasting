from concurrent.futures import ThreadPoolExecutor
import logging.config
import json
import time
import requests
from tqdm import tqdm

# ------------------------------------------------------------------------------#
#                                 INITIALIZE                                    #
# ------------------------------------------------------------------------------#
REST_API_URI = "http://127.0.0.1:5000/predict"
TS_TO_MATCH = [16768, 15735, 16048, 18946, 19536, 19594, 31500, 30761, 37395,
               46011, 47722, 40644, 46352, 51401, 60968]
MIN_VAL = 1000
MAX_VAL = 300000

# Set up logging
logging.basicConfig(level=logging.INFO, format='{"levelname": %(levelname)s, "name": %(name)s,'
                                               ' "message": %(message)s}')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------#
#                                 LOAD TESTING                                  #
# ------------------------------------------------------------------------------#
# Load the input and construct the payload for the request
payload = {'ts_to_match': TS_TO_MATCH, 'min_val': MIN_VAL, 'max_val': MAX_VAL}

def post_url(url):
    return requests.post(url, data=json.dumps(payload),
                         headers={'Content-type': 'application/json'}).json()

def load_test(num_times, num_workers = 3, uri = REST_API_URI):
    start_time = time.time()
    
    list_of_urls = [uri] * num_times
    with ThreadPoolExecutor(max_workers = num_workers) as pool:
        post_request_results = list(tqdm(pool.map(post_url, list_of_urls)))
        
    end_time = time.time()
    time_per_request = (end_time - start_time) / num_times
    logger.info(json.dumps({'time_per_request': time_per_request,
                            'request_results':post_request_results[0]}))


if __name__ == "__main__":
    load_test(100)
